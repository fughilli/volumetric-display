"""Add totem_image_url to spiritmodel.

Revision ID: 20251218_01
Revises: 20251125_06
Create Date: 2025-12-18
"""

import sqlalchemy as sa
from alembic import op

revision = "20251218_01"
down_revision = "20251125_06"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("spiritmodel", sa.Column("totem_image_url", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("spiritmodel", "totem_image_url")
