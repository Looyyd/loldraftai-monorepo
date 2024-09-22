-- AlterTable
ALTER TABLE "Match" ADD COLUMN     "processingErrored" BOOLEAN NOT NULL DEFAULT false;

-- CreateIndex
CREATE INDEX "Match_processed_processingErrored_idx" ON "Match"("processed", "processingErrored");
