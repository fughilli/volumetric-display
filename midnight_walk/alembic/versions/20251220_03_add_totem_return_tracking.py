"""Add totem return tracking table.

Revision ID: 20251220_03
Revises: 20251220_02
Create Date: 2025-12-20
"""

import sqlalchemy as sa
from alembic import op

revision = "20251220_03"
down_revision = "20251220_02"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "totemreturnmodel",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("player_id", sa.String(), nullable=False),
        sa.Column("beacon_id", sa.String(), sa.ForeignKey("beaconmodel.id"), nullable=False),
        sa.Column("spirit_id", sa.String(), sa.ForeignKey("spiritmodel.id"), nullable=False),
        sa.Column("returned_at", sa.Float(), nullable=False),
        sa.Column("new_latitude", sa.Float(), nullable=False),
        sa.Column("new_longitude", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("totemreturnmodel")
