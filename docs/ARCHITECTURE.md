# Target architecture

```text
                    multiple virtual robots
                  R01   R02   R03   R04 ...
                    \    |     |    /
                     \   |     |   /
                      v  v     v  v
                  +-------------------+
                  |  Cloud API        |
                  |  POST /telemetry  |
                  |  POST /predict    |
                  +---------+---------+
                            |
              +-------------+-------------+
              |                           |
              v                           v
       persistent storage             ML artifact
        telemetry/history            model.joblib
              |                           |
              +-------------+-------------+
                            |
                            v
                  +-------------------+
                  | Dashboard/Metrics |
                  +-------------------+
```

The API writes telemetry to PostgreSQL when DATABASE_URL is configured. Local runs use a SQLite file.
