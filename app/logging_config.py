import sqlite3
import json
import os
from datetime import datetime


class PredictionLogger:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.getenv("LOG_DB_PATH", "predictions.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_table()

    def _create_table(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                timestamp   TEXT,
                input_data  TEXT,
                prediction  TEXT,
                probability REAL
            )
        """)
        self.conn.commit()

    def log(self, input_data: dict, prediction: str, probability: float) -> None:
        self.conn.execute(
            "INSERT INTO predictions VALUES (?, ?, ?, ?)",
            (
                datetime.now().isoformat(),  
                json.dumps(input_data),
                prediction,
                probability,
            ),
        )
        self.conn.commit()
