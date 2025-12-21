"""Add player name tracking to story events and player model.

Revision ID: 20251220_02
Revises: 20251218_02
Create Date: 2025-12-20
"""

import sqlalchemy as sa
from alembic import op

revision = "20251220_02"
down_revision = "20251218_02"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add player_name column to StoryEventModel
    op.add_column("storyeventmodel", sa.Column("player_name", sa.String(), nullable=True))

    # Create PlayerModel table
    op.create_table(
        "playermodel",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("encountered_spirits", sa.Text(), nullable=True),
        sa.Column("vanquished_spirits", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("playermodel")
    op.drop_column("storyeventmodel", "player_name")
