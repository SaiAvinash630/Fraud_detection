from app import db, create_app
from app.models import Order, OrderItem

app = create_app()
with app.app_context():
    # First delete all OrderItems (if you have this table)
    OrderItem.query.delete()
    # Then delete all Orders
    Order.query.delete()
    db.session.commit()
    print("All orders and order items deleted.")