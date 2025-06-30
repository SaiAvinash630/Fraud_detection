"""
routes.py

Main routing module for the Flask eCommerce app.
Handles user authentication, product display, cart operations, wishlist,
checkout flow, review system, order history, admin controls, and profile updates.
"""

from datetime import datetime, timedelta
from flask import Blueprint, render_template, redirect, url_for, flash, request, g
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from . import db, login_manager
from .models import User, Product, CartItem, Order, OrderItem, WishlistItem, Review
from .forms import (
    RegisterForm,
    LoginForm,
    ProductForm,
    CheckoutForm,
    QuantityForm,
    ProfileForm,
    ReviewForm,
)
import stripe
from config import Config
from sqlalchemy import or_
import joblib
import pandas as pd
import numpy as np

main_bp = Blueprint("main_bp", __name__)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@main_bp.before_app_request
def load_cart_count():
    if current_user.is_authenticated:
        cart_items = CartItem.query.filter_by(user_id=current_user.id).all()
        g.cart_count = sum(item.quantity for item in cart_items)
    else:
        g.cart_count = 0


# -------------------------------
# Homepage & Product Browsing
# -------------------------------


@main_bp.route("/")
def index():
    search_query = request.args.get("search", "")
    selected_category = request.args.get("category", "")
    sort_option = request.args.get("sort", "")

    products = Product.query

    if search_query:
        products = products.filter(
            or_(
                Product.name.ilike(f"%{search_query}%"),
                Product.description.ilike(f"%{search_query}%"),
            )
        )

    if selected_category:
        products = products.filter(Product.category.ilike(f"%{selected_category}%"))

    if sort_option == "name_asc":
        products = products.order_by(Product.name.asc())
    elif sort_option == "name_desc":
        products = products.order_by(Product.name.desc())
    elif sort_option == "price_asc":
        products = products.order_by(Product.price.asc())
    elif sort_option == "price_desc":
        products = products.order_by(Product.price.desc())

    products = products.all()
    categories = [c[0] for c in db.session.query(Product.category).distinct() if c[0]]

    return render_template(
        "index.html",
        products=products,
        categories=categories,
        search_query=search_query,
        selected_category=selected_category,
        sort_option=sort_option,
    )


# -------------------------------
# Product Detail & Reviews
# -------------------------------
@main_bp.route("/admin/edit_product/<int:product_id>", methods=["GET", "POST"])
@login_required
def edit_product(product_id):
    if not current_user.is_admin:
        flash("Access denied.")
        return redirect(url_for("main_bp.admin"))
    product = Product.query.get_or_404(product_id)
    form = ProductForm(obj=product)
    if form.validate_on_submit():
        product.name = form.name.data
        product.description = form.description.data
        product.price = form.price.data
        product.image_url = form.image_url.data
        product.category = form.category.data
        product.stock = form.stock.data
        db.session.commit()
        flash("Product updated successfully!", "success")
        return redirect(url_for("main_bp.admin"))
    return render_template("edit_product.html", form=form, product=product)


@main_bp.route("/admin/delete_product/<int:product_id>", methods=["POST"])
@login_required
def delete_product(product_id):
    if not current_user.is_admin:
        flash("Access denied.")
        return redirect(url_for("main_bp.admin"))
    product = Product.query.get_or_404(product_id)
    db.session.delete(product)
    db.session.commit()
    flash("Product deleted successfully!", "success")
    return redirect(url_for("main_bp.admin"))


@main_bp.route("/product/<int:product_id>", methods=["GET", "POST"])
def product_detail(product_id):
    """
    Show details of a single product and handle review submissions.
    """
    product = Product.query.get_or_404(product_id)
    review_form = ReviewForm()

    if current_user.is_authenticated and review_form.validate_on_submit():
        new_review = Review(
            user_id=current_user.id,
            product_id=product.id,
            rating=review_form.rating.data,
            comment=review_form.content.data,
        )
        db.session.add(new_review)
        db.session.commit()
        flash("Review submitted!")
        return redirect(url_for("main_bp.product_detail", product_id=product.id))

    return render_template("product.html", product=product, review_form=review_form)


# -------------------------------
# Cart Management
# -------------------------------


@main_bp.route("/add_to_cart/<int:product_id>")
@login_required
def add_to_cart(product_id):
    product = Product.query.get_or_404(product_id)

    if product.stock <= 0:
        flash("This product is out of stock.")
        return redirect(url_for("main_bp.product_detail", product_id=product.id))

    cart_item = CartItem.query.filter_by(
        user_id=current_user.id, product_id=product.id
    ).first()
    if cart_item:
        if cart_item.quantity < product.stock:
            cart_item.quantity += 1
        else:
            flash("Reached available stock limit.")
            return redirect(url_for("main_bp.cart"))
    else:
        cart_item = CartItem(user_id=current_user.id, product_id=product.id, quantity=1)
        db.session.add(cart_item)

    db.session.commit()
    flash(f"Added {product.name} to cart.")
    return redirect(url_for("main_bp.product_detail", product_id=product.id))


@main_bp.route("/cart", methods=["GET", "POST"])
@login_required
def cart():
    cart_items = CartItem.query.filter_by(user_id=current_user.id).all()
    total = sum(item.product.price * item.quantity for item in cart_items)
    form = CheckoutForm()
    quantity_forms = {item.id: QuantityForm(obj=item) for item in cart_items}

    return render_template(
        "cart.html",
        cart_items=cart_items,
        total=total,
        form=form,
        quantity_forms=quantity_forms,
    )


@main_bp.route("/update_cart/<int:item_id>", methods=["POST"])
@login_required
def update_cart(item_id):
    item = CartItem.query.get_or_404(item_id)
    form = QuantityForm()

    if item.user_id != current_user.id:
        flash("Unauthorized.")
        return redirect(url_for("main_bp.cart"))

    if form.validate_on_submit():
        quantity = form.quantity.data
        if quantity > item.product.stock:
            flash("Not enough stock available.")
        else:
            item.quantity = quantity
            db.session.commit()
            flash("Quantity updated.")

    return redirect(url_for("main_bp.cart"))


@main_bp.route("/remove_from_cart/<int:item_id>")
@login_required
def remove_from_cart(item_id):
    item = CartItem.query.get_or_404(item_id)
    if item.user_id != current_user.id:
        flash("Unauthorized.")
        return redirect(url_for("main_bp.cart"))

    db.session.delete(item)
    db.session.commit()
    flash("Item removed.")
    return redirect(url_for("main_bp.cart"))


# -------------------------------
# Checkout & Stripe Integration
# -------------------------------

import pandas as pd
import numpy as np
import joblib
import shap


# Define the FraudPredictor class (must match how you saved the model)
class FraudPredictor:
    def __init__(self, model_path):
        self.model_data = joblib.load(model_path)
        self.xgb_model = self.model_data["xgb_model"]
        self.iso_model = self.model_data["iso_model"]
        self.preprocessor = self.model_data["preprocessor"]
        self.iso_thresh = self.model_data["iso_thresh"]
        self.feature_names = self.model_data["feature_names"]
        self.xgb_explainer = shap.TreeExplainer(self.xgb_model)

    def predict(self, transaction_data):
        processed = self.preprocessor.transform(pd.DataFrame([transaction_data]))
        xgb_prob = self.xgb_model.predict_proba(processed)[0][1]
        iso_score = self.iso_model.decision_function(processed)[0]
        if xgb_prob > 0.8:
            decision = "FRAUD"
        elif (xgb_prob > 0.7) or ((xgb_prob > 0.4) and (iso_score < self.iso_thresh)):
            decision = "NEED TO TAKE FEEDBACK"
        else:
            decision = "GENUINE"
        shap_values = self.xgb_explainer.shap_values(processed)[0]
        indicators = []
        total_impact = sum(np.abs(shap_values))
        for i, val in enumerate(shap_values):
            if val > 0:
                impact_percent = (abs(val) / total_impact) * 100
                indicators.append(
                    {
                        "feature": self.feature_names[i],
                        "value": processed[0][i],
                        "impact_percent": round(impact_percent, 1),
                    }
                )
        indicators = sorted(
            indicators, key=lambda x: x["impact_percent"], reverse=True
        )[:5]
        return {
            "decision": decision,
            "probability": round(float(xgb_prob), 4),
            "anomaly_score": round(float(iso_score), 4),
            "fraud_indicators": indicators,
            "thresholds": {
                "xgb_high": 0.7,
                "xgb_medium": 0.4,
                "iso_threshold": round(float(self.iso_thresh), 4),
            },
        }


@main_bp.route("/checkout", methods=["POST"])
@login_required
def checkout():
    form = CheckoutForm()

    if form.validate_on_submit():
        cart_items = CartItem.query.filter_by(user_id=current_user.id).all()

        if not cart_items:
            flash("Your cart is empty.")
            return redirect(url_for("main_bp.cart"))

        for item in cart_items:
            if item.quantity > item.product.stock:
                flash(f"Not enough stock for {item.product.name}.")
                return redirect(url_for("main_bp.cart"))

        # Prepare transaction features
        total_amount = sum(item.product.price * item.quantity for item in cart_items)
        account_age_days = (
            (datetime.now() - current_user.created_at).days
            if current_user.created_at
            else 0
        )
        device = request.user_agent.platform or "Mobile"
        transaction_hour = datetime.now().hour
        category = cart_items[0].product.category if cart_items else "Unknown"
        transaction_time = datetime.now()

        time_24h_ago = transaction_time - timedelta(hours=24)
        orders_last_24h = Order.query.filter(
            Order.user_id == current_user.id,
            Order.timestamp >= time_24h_ago,
            Order.timestamp <= transaction_time,
        ).all()

        num_trans_24h = sum(1 for o in orders_last_24h if o.status == "Completed")
        num_failed_24h = sum(1 for o in orders_last_24h if o.status == "Cancelled")

        sample_transaction = {
            "User_id": current_user.id,
            "account_age_days": account_age_days,
            "payment_method": "Credit Card",
            "device": device,
            "category": category,
            "amount": total_amount,
            "num_trans_24h": num_trans_24h,
            "num_failed_24h": num_failed_24h,
            "no_of_cards_from_ip": getattr(current_user, "no_of_cards_from_ip", 0),
            "transaction_hour": transaction_hour,
        }

        # Add engineered features
        sample_transaction["amount_log"] = np.log1p(sample_transaction["amount"])
        sample_transaction["hour_sin"] = np.sin(
            2 * np.pi * sample_transaction["transaction_hour"] / 24
        )
        sample_transaction["hour_cos"] = np.cos(
            2 * np.pi * sample_transaction["transaction_hour"] / 24
        )

        user_orders = (
            Order.query.filter_by(user_id=current_user.id)
            .order_by(Order.timestamp)
            .all()
        )

        user_avg = (
            db.session.query(db.func.avg(Order.total_amount))
            .filter(Order.user_id == current_user.id)
            .scalar()
        ) or 1.0
        sample_transaction["amount_to_avg"] = sample_transaction["amount"] / user_avg

        last_order = (
            Order.query.filter_by(user_id=current_user.id)
            .order_by(Order.timestamp.desc())
            .first()
        )

        last_device = last_order.device if last_order else device
        sample_transaction["new_device_flag"] = int(
            sample_transaction["device"] != last_device
        )

        # Remove User_id for prediction
        transaction_input = sample_transaction.copy()
        transaction_input.pop("User_id", None)

        # Load model and predict
        fraud_predictor = FraudPredictor("models/hybrid_fraud_model_3_outputs.pkl")
        prediction = fraud_predictor.predict(transaction_input)

        if prediction["decision"] != "GENUINE":
            flash(
                "Transaction flagged as potentially fraudulent. Please contact support."
            )
            order = Order(
                user_id=current_user.id,
                total_amount=total_amount,
                payment_method="Credit Card",
                device=device,
                status="Cancelled",
            )
            db.session.add(order)
            db.session.commit()
            return redirect(url_for("main_bp.cart"))

        try:
            flash("Simulated payment successful! Redirecting to order completion.")
            return redirect(url_for("main_bp.checkout_success"))
        except Exception:
            flash("Payment failed. Please try again.")
            return redirect(url_for("main_bp.cart"))

    flash("Invalid checkout form.")
    return redirect(url_for("main_bp.cart"))


@main_bp.route("/checkout/success")
@login_required
def checkout_success():
    cart_items = CartItem.query.filter_by(user_id=current_user.id).all()
    total_amount = sum(item.product.price * item.quantity for item in cart_items)

    device = request.user_agent.platform or "Unknown"
    # transaction_time = datetime.now()
    payment_method = "Credit Card"  # Placeholder, update as needed

    order = Order(
        user_id=current_user.id,
        total_amount=total_amount,
        payment_method=payment_method,
        device=device,
        # transaction_time=transaction_time,
        status="Completed",
    )
    db.session.add(order)
    db.session.commit()

    for item in cart_items:
        order_item = OrderItem(
            order_id=order.id, product_id=item.product.id, quantity=item.quantity
        )
        item.product.stock -= item.quantity
        db.session.add(order_item)

    CartItem.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()

    flash("Payment successful! Thank you for your purchase.")
    return redirect(url_for("main_bp.index"))


# -------------------------------
# Order History
# -------------------------------


@main_bp.route("/orders")
@login_required
def orders():
    user_orders = (
        Order.query.filter_by(user_id=current_user.id)
        .order_by(Order.timestamp.desc())
        .all()
    )
    return render_template("orders.html", orders=user_orders)


# -------------------------------
# User Profile Page
# -------------------------------


@main_bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    form = ProfileForm(obj=current_user)

    if form.validate_on_submit():
        current_user.address = form.address.data
        current_user.preferences = form.preferences.data
        db.session.commit()
        flash("Profile updated successfully.")
        return redirect(url_for("main_bp.profile"))

    return render_template("profile.html", form=form)


# -------------------------------
# Wishlist
# -------------------------------


@main_bp.route("/wishlist")
@login_required
def wishlist():
    wishlist = WishlistItem.query.filter_by(user_id=current_user.id).all()
    return render_template("wishlist.html", wishlist=wishlist)


@main_bp.route("/add_to_wishlist/<int:product_id>")
@login_required
def add_to_wishlist(product_id):
    product = Product.query.get_or_404(product_id)
    existing = WishlistItem.query.filter_by(
        user_id=current_user.id, product_id=product.id
    ).first()
    if existing:
        flash("Product already in your wishlist.")
    else:
        wishlist_item = WishlistItem(user_id=current_user.id, product_id=product.id)
        db.session.add(wishlist_item)
        db.session.commit()
        flash("Product added to your wishlist.")
    return redirect(url_for("main_bp.product_detail", product_id=product.id))


@main_bp.route("/remove_from_wishlist/<int:product_id>")
@login_required
def remove_from_wishlist(product_id):
    item = WishlistItem.query.filter_by(
        user_id=current_user.id, product_id=product_id
    ).first()
    if item:
        db.session.delete(item)
        db.session.commit()
        flash("Removed from wishlist.")
    else:
        flash("Item not found in wishlist.")
    return redirect(request.referrer or url_for("main_bp.index"))


# -------------------------------
# User Authentication
# -------------------------------


@main_bp.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            flash("Logged in successfully.")
            return redirect(url_for("main_bp.index"))
        flash("Invalid email or password.")
    return render_template("login.html", form=form)


@main_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.")
    return redirect(url_for("main_bp.index"))


@main_bp.route("/register", methods=["GET", "POST"])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        existing_user = User.query.filter_by(email=form.email.data).first()
        if existing_user:
            flash("Email already registered.")
            return redirect(url_for("main_bp.register"))

        hashed_password = generate_password_hash(form.password.data)
        user = User(
            email=form.email.data,
            password=hashed_password,
            created_at=datetime.now(),
        )
        db.session.add(user)
        db.session.commit()
        flash("Registration successful. Please log in.")
        return redirect(url_for("main_bp.login"))

    return render_template("register.html", form=form)


@main_bp.route("/admin", methods=["GET", "POST"])
@login_required
def admin():
    if not current_user.is_admin:
        flash("Access denied.")
        return redirect(url_for("main_bp.index"))
    form = ProductForm()
    if form.validate_on_submit():
        new_product = Product(
            name=form.name.data,
            description=form.description.data,
            price=form.price.data,
            image_url=form.image_url.data,
            category=form.category.data,
            stock=form.stock.data,
        )
        db.session.add(new_product)
        db.session.commit()
        flash("Product added successfully!", "success")
        return redirect(url_for("main_bp.admin"))

    products = Product.query.all()
    return render_template("admin.html", products=products, form=form)
