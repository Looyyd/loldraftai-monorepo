module.exports = {
  apps: [
    {
      name: "extract-to-azure",
      script: "./extractToAzure.sh",
      interpreter: "/bin/bash",
      cwd: "/home/azureuser/draftking-monorepo/apps/data-collection",
      autorestart: true,
      max_restarts: 10,
      restart_delay: 4000,
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      error_file: "./logs/extractToAzure_error.log",
      out_file: "./logs/extractToAzure_out.log",
    },
    ...["EUW1", "KR"].flatMap((region, regionIndex) =>
      ["collectMatchIds", "fetchPuuids", "processMatches", "updateLadder"].map(
        (script) => ({
          name: `${script}-${region}`,
          script:
            script === "updateLadder"
              ? `sleep ${
                  regionIndex * 1800
                } && yarn tsx ./src/scripts/${script}.ts --region ${region}`
              : `yarn tsx ./src/scripts/${script}.ts --region ${region}`,
          autorestart: true,
          max_restarts: 10,
          restart_delay: 4000,
          log_date_format: "YYYY-MM-DD HH:mm:ss",
          error_file: `./logs/${script}_${region}_error.log`,
          out_file: `./logs/${script}_${region}_out.log`,
        })
      )
    ),
  ],
};
