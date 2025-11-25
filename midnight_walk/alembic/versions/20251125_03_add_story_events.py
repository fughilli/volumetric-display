"""Add story event log table.

Revision ID: 20251125_03
Revises: 20251125_02
Create Date: 2025-11-25
"""

import sqlalchemy as sa
from alembic import op

revision = "20251125_03"
down_revision = "20251125_02"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "storyeventmodel",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("beacon_id", sa.String(), sa.ForeignKey("beaconmodel.id"), nullable=True),
        sa.Column("spirit_id", sa.String(), sa.ForeignKey("spiritmodel.id"), nullable=True),
        sa.Column("spirit_name", sa.String(), nullable=True),
        sa.Column("message", sa.Text(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("storyeventmodel")
