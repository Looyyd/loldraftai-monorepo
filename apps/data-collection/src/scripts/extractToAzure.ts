// src/scripts/extractToAzure.ts
import fs from "fs";
import { BlobServiceClient, ContainerClient } from "@azure/storage-blob";
import { Match, PrismaClient } from "@prisma/client";
import { spawn } from "child_process";
import path from "path";
import { config } from "dotenv";
import { telemetry } from "../utils/telemetry";
import {
  DatabaseBackoff,
  LoggerFunction,
} from "../utils/databaseErrorHandling";

config();

const AZURE_CONNECTION_STRING = process.env.AZURE_CONNECTION_STRING!;

if (!AZURE_CONNECTION_STRING) {
  throw new Error("AZURE_CONNECTION_STRING is not set");
}

const BATCH_SIZE = 512 * 4;

const CONTAINER_NAME = "league-matches";
const RAW_DATA_PREFIX = "raw";
const PROCESSED_DATA_PREFIX = "processed";
const currentFileDir = path.dirname(new URL(import.meta.url).pathname);

interface ExtractorConfig {
  batchSize: number;
  tempDir: string;
}

interface FileSaver {
  saveFile(fileName: string, data: string | Buffer): Promise<void>;
}

class AzureFileSaver implements FileSaver {
  constructor(private containerClient: ContainerClient) {}

  async saveFile(fileName: string, data: string | Buffer): Promise<void> {
    const blob = this.containerClient.getBlockBlobClient(fileName);
    if (typeof data === "string") {
      await blob.upload(data, data.length);
    } else {
      await blob.uploadData(data);
    }
  }
}

class LocalFileSaver implements FileSaver {
  private tempDir: string;

  constructor() {
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    this.tempDir = `/tmp/${timestamp}`;
    fs.mkdirSync(this.tempDir, { recursive: true });
  }

  async saveFile(fileName: string, data: string | Buffer): Promise<void> {
    const filePath = path.join(this.tempDir, fileName);
    const dir = path.dirname(filePath);
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(filePath, data);
  }
}

// Add a more comprehensive logger at the top of the file
const log = (level: "INFO" | "ERROR" | "DEBUG", message: string) => {
  // console.log(`[${new Date().toISOString()}] [${level}] ${message}`);
};

class MatchExtractor {
  private prisma: PrismaClient;
  private remoteFileSaver: FileSaver;
  private dbBackoff = new DatabaseBackoff();

  constructor(private config: ExtractorConfig) {
    log(
      "INFO",
      `Initializing MatchExtractor with batch size: ${config.batchSize}`
    );
    this.prisma = new PrismaClient();
    const blobServiceClient = BlobServiceClient.fromConnectionString(
      AZURE_CONNECTION_STRING
    );
    const containerClient =
      blobServiceClient.getContainerClient(CONTAINER_NAME);

    this.remoteFileSaver =
      process.env.NODE_ENV === "development"
        ? new LocalFileSaver()
        : new AzureFileSaver(containerClient);

    log(
      "INFO",
      `Using ${
        process.env.NODE_ENV === "development"
          ? "LocalFileSaver"
          : "AzureFileSaver"
      }`
    );
  }

  async initialize() {
    log("DEBUG", `Creating temp directory: ${this.config.tempDir}`);
    fs.mkdirSync(this.config.tempDir, { recursive: true });
    log("INFO", "Initialization complete");
  }

  async extractBatch() {
    try {
      log("DEBUG", "Starting new batch extraction");

      // Get batch of unprocessed matches with retry
      log(
        "DEBUG",
        `Querying for up to ${this.config.batchSize} unprocessed matches`
      );
      const matches = await this.dbBackoff.withRetry(async () => {
        return this.prisma.$queryRaw`
          SELECT *
          FROM "Match"
          WHERE processed = true
          AND exported = false
          AND "processingErrored" = false
          ORDER BY processed, exported, "processingErrored"
          LIMIT ${this.config.batchSize}
        ` as Promise<Match[]>;
      });

      log("INFO", `Found ${matches.length} matches to process`);

      // Process full batches only
      if (matches.length < this.config.batchSize) {
        log(
          "INFO",
          `Incomplete batch (${matches.length} < ${this.config.batchSize}), skipping`
        );
        return 0;
      }

      const now = new Date();
      const year = now.getFullYear();
      const month = (now.getMonth() + 1).toString().padStart(2, "0");
      const batchId = now.toISOString().replace(/[:.]/g, "-");

      const processedFileName = `${PROCESSED_DATA_PREFIX}/${year}/${month}/${batchId}.parquet`;

      const rawLocalFilePath = path.join(
        this.config.tempDir,
        `raw_${batchId}.json`
      );
      const processedLocalFilePath = path.join(
        this.config.tempDir,
        `processed_${batchId}.parquet`
      );

      log("DEBUG", `Writing temporary raw file to: ${rawLocalFilePath}`);
      fs.mkdirSync(path.dirname(rawLocalFilePath), { recursive: true });
      fs.writeFileSync(rawLocalFilePath, JSON.stringify(matches));

      log("DEBUG", "Spawning Python process for Parquet conversion");
      const pythonProcess = spawn("python", [
        path.join(currentFileDir, "createParquet.py"),
        "--batch-file",
        rawLocalFilePath,
        "--output-file",
        processedLocalFilePath,
      ]);

      let pythonOutput = "";
      let pythonError = "";

      pythonProcess.stdout.on("data", (data) => {
        pythonOutput += data.toString();
        log("DEBUG", `Python stdout: ${data.toString().trim()}`);
      });

      pythonProcess.stderr.on("data", (data) => {
        pythonError += data.toString();
        log("ERROR", `Python stderr: ${data.toString().trim()}`);
      });

      await new Promise((resolve, reject) => {
        pythonProcess.on("exit", (code) => {
          log("DEBUG", `Python process exited with code ${code}`);
          if (code === 0) {
            resolve(null);
          } else {
            reject(
              new Error(
                `Python process exited with code ${code}.\nError: ${pythonError}\nOutput: ${pythonOutput}`
              )
            );
          }
        });
      });

      log("DEBUG", `Reading processed Parquet file: ${processedLocalFilePath}`);
      const processedData = fs.readFileSync(processedLocalFilePath);

      log("DEBUG", `Saving processed data to: ${processedFileName}`);
      await this.remoteFileSaver.saveFile(processedFileName, processedData);

      log("DEBUG", "Updating match records in database");
      await this.dbBackoff.withRetry(async () => {
        await this.prisma.match.updateMany({
          where: {
            id: {
              in: matches.map((m) => m.id),
            },
          },
          data: {
            exported: true,
          },
        });
      });

      log("DEBUG", "Cleaning up temporary files");
      fs.rmSync(this.config.tempDir, { recursive: true, force: true });

      log("INFO", `Successfully processed ${matches.length} matches`);
      telemetry.trackEvent("MatchesExtracted", {
        count: matches.length,
      });

      if (matches.length > 0) {
        log("DEBUG", "Adding delay between batches (100ms)");
        await new Promise((resolve) => setTimeout(resolve, 100));
      }

      return matches.length;
    } catch (error) {
      log("ERROR", `Batch extraction failed: ${error}`);
      throw error;
    }
  }

  async run() {
    try {
      log("INFO", "Starting extractor run");
      await this.initialize();

      // Get the maximum runtime in minutes for more precise control
      const maxRuntimeMinutes = parseInt(
        process.env.MAX_RUNTIME_MINUTES || "355"
      );
      const startTime = Date.now();
      const endTime = startTime + maxRuntimeMinutes * 60 * 1000;

      log(
        "INFO",
        `Will run for ${maxRuntimeMinutes} minutes (${(
          maxRuntimeMinutes / 60
        ).toFixed(2)} hours) until ${new Date(endTime).toISOString()}`
      );

      while (Date.now() < endTime) {
        const processedCount = await this.extractBatch();

        if (processedCount === 0) {
          // Calculate time left
          const timeLeftMs = endTime - Date.now();
          // If less than 15 minutes left, sleep for a shorter time
          const sleepTimeMs = Math.min(15 * 60 * 1000, timeLeftMs);

          if (sleepTimeMs > 0) {
            log(
              "INFO",
              `No matches to process, sleeping for ${Math.round(
                sleepTimeMs / 1000 / 60
              )} minutes`
            );
            await new Promise((resolve) => setTimeout(resolve, sleepTimeMs));
          }
        }
      }

      log(
        "INFO",
        `Reached maximum runtime of ${maxRuntimeMinutes} minutes, exiting`
      );
    } catch (error) {
      log("ERROR", `Extractor crashed: ${error}`);
      await telemetry.flush();
      throw error;
    }
  }
}

// Run the extractor
const extractor = new MatchExtractor({
  batchSize: BATCH_SIZE,
  tempDir: "/tmp/leaguedraft-extractor",
});

extractor.run().catch(console.error);
