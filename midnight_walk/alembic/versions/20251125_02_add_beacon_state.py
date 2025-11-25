"""Add beacon state column.

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
        sa.Column("state", sa.Text(), nullable=False, server_default="undiscovered"),
    )
    op.execute(sa.text("UPDATE beaconmodel SET state = COALESCE(state, 'undiscovered')"))


def downgrade() -> None:
    op.drop_column("beaconmodel", "state")
