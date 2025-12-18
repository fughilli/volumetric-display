"""Add minigame table.

Revision ID: 20251218_02
Revises: 20251218_01
Create Date: 2025-12-18
"""

import sqlalchemy as sa
from alembic import op

revision = "20251218_02"
down_revision = "20251218_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "minigamemodel",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("spirit_id", sa.String(), nullable=False),
        sa.Column("html_content", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["spirit_id"], ["spiritmodel.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("spirit_id"),
    )


def downgrade() -> None:
    op.drop_table("minigamemodel")
