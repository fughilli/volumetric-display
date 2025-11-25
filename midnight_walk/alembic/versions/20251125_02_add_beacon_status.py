"""Add beacon status column.

Revision ID: 20251125_02
Revises: 20251125_01
Create Date: 2025-11-25
"""

import sqlalchemy as sa
from alembic import op

revision = "20251125_02"
down_revision = "20251125_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "beaconmodel",
        sa.Column("status", sa.Text(), nullable=False, server_default="unknown"),
    )
    op.alter_column("beaconmodel", "status", server_default=None)


def downgrade() -> None:
    op.drop_column("beaconmodel", "status")
