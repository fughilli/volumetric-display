"""Consolidate 'returned' state to 'encounter'.

Revision ID: 20251125_06
Revises: 20251125_05
Create Date: 2025-11-25
"""

import sqlalchemy as sa
from alembic import op

revision = "20251125_06"
down_revision = "20251125_05"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Convert all "returned" states to "encounter"
    op.execute(sa.text("UPDATE beaconmodel SET state = 'encounter' WHERE state = 'returned'"))


def downgrade() -> None:
    # Convert "encounter" back to "returned" (if needed for rollback)
    # Note: This is a lossy operation - we can't distinguish which encounters were "returned"
    op.execute(sa.text("UPDATE beaconmodel SET state = 'returned' WHERE state = 'encounter'"))
