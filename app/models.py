"""
models.py

Defines all SQLAlchemy models for the eCommerce web app:
User accounts, Products, Cart, Orders, Wishlist, and Reviews.
"""

from flask_login import UserMixin
from datetime import datetime
from app import db


class User(UserMixin, db.Model):
    """
    User account model.
    Includes relationships for wishlist, cart, reviews, and orders.
    """

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)

    # Optional profile data
    address = db.Column(db.String(300))
    preferences = db.Column(db.Text)

    # New fields
    created_at = db.Column(db.DateTime, default=datetime.now())  # Registration date
    # Relationships
    wishlist = db.relationship(
        "WishlistItem", backref="user", cascade="all, delete-orphan"
    )
    reviews = db.relationship("Review", backref="user", cascade="all, delete-orphan")
    cart_items = db.relationship(
        "CartItem", backref="user", cascade="all, delete-orphan"
    )
    orders = db.relationship("Order", backref="user", cascade="all, delete-orphan")


class Product(db.Model):
    """
    Product model for items in the store.
    Includes category, price, stock, and image URL.
    """

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    description = db.Column(db.Text)
    price = db.Column(db.Float, nullable=False)
    image_url = db.Column(db.String(300))
    category = db.Column(db.String(100))
    stock = db.Column(db.Integer, default=10)

    # Relationships
    reviews = db.relationship("Review", backref="product", cascade="all, delete-orphan")
    wishlist_items = db.relationship(
        "WishlistItem", backref="product", cascade="all, delete-orphan"
    )
    order_items = db.relationship(
        "OrderItem", backref="product", cascade="all, delete-orphan"
    )


class CartItem(db.Model):
    """
    Represents an item in a user's shopping cart.
    """

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    product_id = db.Column(db.Integer, db.ForeignKey("product.id"))
    quantity = db.Column(db.Integer, default=1)

    # Direct product reference for easier access
    product = db.relationship("Product")


class Order(db.Model):
    """
    Represents a placed order by a user.
    Includes total amount and timestamp.
    """

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    total_amount = db.Column(db.Float)
    payment_method = db.Column(db.String(50))
    device = db.Column(db.String(50))
    status = db.Column(db.String(50))  # e.g. Completed, Cancelled, Needs Feedback
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    # One-to-many: Order -> OrderItems
    items = db.relationship("OrderItem", backref="order", cascade="all, delete-orphan")


class OrderItem(db.Model):
    """
    Represents a single product within an order.
    """

    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.Integer, db.ForeignKey("order.id"))
    product_id = db.Column(db.Integer, db.ForeignKey("product.id"))
    quantity = db.Column(db.Integer)


class WishlistItem(db.Model):
    """
    Represents a product saved to a user's wishlist.
    """

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    product_id = db.Column(db.Integer, db.ForeignKey("product.id"))


class Review(db.Model):
    """
    Review written by a user for a product.
    Includes star rating, comment, and timestamp.
    """

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    product_id = db.Column(db.Integer, db.ForeignKey("product.id"))
    rating = db.Column(db.Integer)  # e.g. 1–5 stars
    comment = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
