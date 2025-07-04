"""Add FeedbackCase table

Revision ID: 32eee9ce39e5
Revises: dfcfab35041d
Create Date: 2025-06-30 19:47:25.324006

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '32eee9ce39e5'
down_revision = 'dfcfab35041d'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('feedback_case',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('payment_method', sa.String(length=50), nullable=True),
    sa.Column('device', sa.String(length=50), nullable=True),
    sa.Column('category', sa.String(length=50), nullable=True),
    sa.Column('amount', sa.Float(), nullable=True),
    sa.Column('quantity', sa.Integer(), nullable=True),
    sa.Column('total_value', sa.Float(), nullable=True),
    sa.Column('num_trans_24h', sa.Integer(), nullable=True),
    sa.Column('num_failed_24h', sa.Integer(), nullable=True),
    sa.Column('no_of_cards_from_ip', sa.Integer(), nullable=True),
    sa.Column('account_age_days', sa.Integer(), nullable=True),
    sa.Column('timestamp', sa.DateTime(), nullable=True),
    sa.Column('prediction', sa.String(length=50), nullable=True),
    sa.Column('probability', sa.Float(), nullable=True),
    sa.Column('anomaly_score', sa.Float(), nullable=True),
    sa.Column('admin_status', sa.String(length=20), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('feedback_case')
    # ### end Alembic commands ###
