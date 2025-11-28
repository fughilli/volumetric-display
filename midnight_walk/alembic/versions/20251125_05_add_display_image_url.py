"""Add image_url to display location.

Revision ID: 20251125_05
Revises: 20251125_04
Create Date: 2025-11-25
"""

import sqlalchemy as sa
from alembic import op

revision = "20251125_05"
down_revision = "20251125_04"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("displaylocationmodel", sa.Column("image_url", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("displaylocationmodel", "image_url")
