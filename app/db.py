"""Persistent telemetry storage for local and hosted deployments."""
import os
from pathlib import Path

from sqlalchemy import Column, Float, Integer, MetaData, String, Table, create_engine, func, insert, select

default_db = Path(__file__).resolve().parents[1] / "fleet.db"
database_url = os.getenv("DATABASE_URL", f"sqlite:///{default_db.as_posix()}")
if database_url.startswith("postgresql://"):
    database_url = database_url.replace("postgresql://", "postgresql+psycopg://", 1)

engine = create_engine(database_url, pool_pre_ping=True)
metadata = MetaData()
telemetry = Table(
    "telemetry",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("robot_id", String(20), nullable=False, index=True),
    *(
        Column(name, Float, nullable=False)
        for name in (
            "timestamp", "x", "y", "speed", "battery", "motor_temp",
            "vibration", "motor_current", "obstacle_distance",
        )
    ),
)
sample_columns = [column for column in telemetry.c if column.name != "id"]


def init_db():
    metadata.create_all(engine)


def store_sample(sample):
    with engine.begin() as connection:
        connection.execute(insert(telemetry).values(**sample))


def robot_count():
    query = select(func.count(telemetry.c.robot_id.distinct()))
    with engine.connect() as connection:
        return connection.execute(query).scalar_one()


def latest_samples():
    latest_ids = select(func.max(telemetry.c.id)).group_by(telemetry.c.robot_id)
    query = select(*sample_columns).where(telemetry.c.id.in_(latest_ids)).order_by(telemetry.c.id)
    with engine.connect() as connection:
        return [dict(row) for row in connection.execute(query).mappings()]


def history(robot_id):
    query = (
        select(*sample_columns)
        .where(telemetry.c.robot_id == robot_id)
        .order_by(telemetry.c.id)
    )
    with engine.connect() as connection:
        return [dict(row) for row in connection.execute(query).mappings()]
