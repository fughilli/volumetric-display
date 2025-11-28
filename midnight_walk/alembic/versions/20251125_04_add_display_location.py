"""Add display location table.

Revision ID: 20251125_04
Revises: 20251125_03
Create Date: 2025-11-25
"""

import sqlalchemy as sa
from alembic import op

revision = "20251125_04"
down_revision = "20251125_03"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "displaylocationmodel",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("displaylocationmodel")
