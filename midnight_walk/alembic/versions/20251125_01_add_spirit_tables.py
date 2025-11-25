"""Add spirit tables and beacon find log.

Revision ID: 20251125_01
Revises:
Create Date: 2025-11-25
"""

import sqlalchemy as sa
from alembic import op

revision = "20251125_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "spiritmodel",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("blurb", sa.Text(), nullable=False),
        sa.Column("image_url", sa.Text(), nullable=True),
        sa.Column("current_activity", sa.Text(), nullable=True),
    )

    op.create_table(
        "beaconfindmodel",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("beacon_id", sa.Text(), nullable=False),
        sa.Column("found_at", sa.Float(), nullable=False),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column("ip_address", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["beacon_id"], ["beaconmodel.id"], ondelete="CASCADE"),
    )

    op.add_column("beaconmodel", sa.Column("spirit_id", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("beaconmodel", "spirit_id")
    op.drop_table("beaconfindmodel")
    op.drop_table("spiritmodel")
