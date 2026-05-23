"""initial registry schema

Revision ID: 0001
Revises:
Create Date: 2026-05-22 12:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE TYPE deployment_environment AS ENUM ('dev','staging','prod')")
    op.execute("CREATE TYPE deployment_status AS ENUM ('pending','active','failed','rolled_back')")

    op.create_table(
        "experiments",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("name", sa.String, nullable=False, unique=True),
        sa.Column("description", sa.Text),
        sa.Column("owner", sa.String, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "deployments",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("model_version_id", sa.BigInteger, nullable=False),
        sa.Column("environment", postgresql.ENUM("dev", "staging", "prod", name="deployment_environment", create_type=False), nullable=False),
        sa.Column("status", postgresql.ENUM("pending", "active", "failed", "rolled_back", name="deployment_status", create_type=False), nullable=False),
        sa.Column("endpoint_url", sa.String),
        sa.Column("deployed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("retired_at", sa.DateTime(timezone=True)),
    )

    op.create_table(
        "models",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("experiment_id", sa.BigInteger, sa.ForeignKey("experiments.id"), nullable=False),
        sa.Column("name", sa.String, nullable=False, unique=True),
        sa.Column("current_deployment_id", sa.BigInteger, sa.ForeignKey("deployments.id")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "model_versions",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("model_id", sa.BigInteger, sa.ForeignKey("models.id"), nullable=False),
        sa.Column("version_tag", sa.String, nullable=False),
        sa.Column("artifact_uri", sa.String, nullable=False),
        sa.Column("framework", sa.String, nullable=False),
        sa.Column("hyperparameters", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("metrics", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("trained_by", sa.String, nullable=False),
        sa.Column("trained_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("archived_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("model_id", "version_tag", name="uq_modelver_model_version"),
    )
    op.create_index("idx_versions_model_trained_at", "model_versions", ["model_id", "trained_at"])
    op.create_index("idx_versions_metrics_gin", "model_versions", ["metrics"], postgresql_using="gin")
    op.create_index("idx_versions_hyperparams_gin", "model_versions", ["hyperparameters"], postgresql_using="gin")

    op.create_foreign_key(
        "fk_deployments_model_version", "deployments", "model_versions",
        ["model_version_id"], ["id"],
    )

    op.create_table(
        "model_version_lineage",
        sa.Column("child_id", sa.BigInteger, sa.ForeignKey("model_versions.id"), primary_key=True),
        sa.Column("parent_id", sa.BigInteger, sa.ForeignKey("model_versions.id"), primary_key=True),
        sa.Column("edge_type", sa.String, nullable=False),
        sa.CheckConstraint("child_id <> parent_id", name="ck_no_self_lineage"),
    )

    op.execute("""
        CREATE UNIQUE INDEX uniq_one_active_per_model_env
        ON deployments (model_version_id, environment)
        WHERE status = 'active' AND retired_at IS NULL
    """)


def downgrade() -> None:
    op.drop_table("model_version_lineage")
    op.drop_table("model_versions")
    op.drop_table("models")
    op.drop_table("deployments")
    op.drop_table("experiments")
    op.execute("DROP TYPE IF EXISTS deployment_status")
    op.execute("DROP TYPE IF EXISTS deployment_environment")
