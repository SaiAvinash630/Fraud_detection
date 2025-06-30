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
from .models import User, Product, CartItem, Order, OrderItem, WishlistItem, Review, FeedbackCase
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
import shap
from app.feature_engineering import FeatureEngineer
from sklearn.calibration import CalibratedClassifierCV
# Define the FraudPredictor class (must match how you saved the model)
class FraudPredictor:
    def __init__(self, model_path):
        self.model_data = joblib.load(model_path)
        self.xgb_model = self.model_data["xgb_model"]
        self.iso_model = self.model_data["iso_model"]
        self.pipeline = self.model_data["pipeline"]
        self.iso_thresh = self.model_data["iso_thresh"]
        self.feature_names = self.model_data["feature_names"]
        if hasattr(self.xgb_model, "base_estimator_"):
            shap_model = self.xgb_model.base_estimator_
        elif hasattr(self.xgb_model, "estimator"):
            shap_model = self.xgb_model.estimator
        else:
            shap_model = self.xgb_model
        self.xgb_explainer = shap.TreeExplainer(shap_model)

    def predict(self, transaction_data):
        processed = self.pipeline.transform(pd.DataFrame([transaction_data]))
        xgb_prob = self.xgb_model.predict_proba(processed)[0][1]
        iso_score = self.iso_model.decision_function(processed)[0]

        # Decision logic
        if xgb_prob > 0.9:
            decision = "FRAUD"
        # elif xgb_prob > 0.5 and iso_score < self.iso_thresh:
        elif xgb_prob > 0.6 and iso_score < 0.01:
            decision = "NEED TO TAKE FEEDBACK"
        else:
            decision = "GENUINE"

        # SHAP explanation
        shap_values = self.xgb_explainer.shap_values(processed)[0]
        indicators = []
        total_impact = sum(np.abs(shap_values))
        for i, val in enumerate(shap_values):
            indicators.append(
                {
                    "feature": (
                        self.feature_names[i]
                        if i < len(self.feature_names)
                        else f"feature_{i}"
                    ),
                    "value": float(processed[0][i]),
                    "impact_percent": (
                        round((abs(val) / total_impact) * 100, 2)
                        if total_impact
                        else 0.0
                    ),
                }
            )
        indicators = sorted(
            indicators, key=lambda x: x["impact_percent"], reverse=True
        )[:5]

        fraud_pattern = indicators[0]["feature"] if indicators else "unknown"

        return {
            "decision": decision,
            "probability": round(float(xgb_prob), 4),
            "anomaly_score": round(float(iso_score), 4),
            "fraud_indicators": indicators,
            "fraud_pattern": fraud_pattern,
            "thresholds": {
                "xgb_high": 0.9,
                "xgb_feedback": 0.6,
                "iso_threshold": round(float(self.iso_thresh), 4),
            },
        }


fraud_predictor = FraudPredictor("models/hybrid.pkl")

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
    selected_category = request.args.get("category", "")

    # Get all unique categories for the homepage
    categories = [c[0] for c in db.session.query(Product.category).distinct() if c[0]]

    if not selected_category:
        # No category selected: show only categories
        return render_template(
            "index.html",
            categories=categories,
            selected_category=None,
            products=None,
        )
    else:
        # Category selected: show products in that category
        products = Product.query.filter_by(category=selected_category).all()
        return render_template(
            "index.html",
            categories=categories,
            selected_category=selected_category,
            products=products,
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

        # All checks passed, redirect to payment page
        return redirect(url_for("main_bp.payment"))

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


@main_bp.route('/payment', methods=['GET', 'POST'])
@login_required
def payment():
    account_age_days = (datetime.now() - current_user.created_at).days if current_user.created_at else 0
    allowed_methods = ['Credit Card', 'Debit Card', 'Wallet', 'Net Banking']

    if request.method == 'POST':
        payment_method = request.form.get('payment_method')
        cart_items = CartItem.query.filter_by(user_id=current_user.id).all()
        # Filter out cart items with missing products
        cart_items = [item for item in cart_items if item.product is not None]

        if payment_method not in allowed_methods:
            flash("Please select a valid payment method.")
            return render_template('payment.html')

        if not cart_items:
            flash("Your cart is empty or contains unavailable products.")
            return redirect(url_for("main_bp.cart"))

        # Safely get category and product_id
        first_product = cart_items[0].product if cart_items else None
        category = first_product.category if first_product else "Unknown"
        product_id = first_product.id if first_product else None

        total_amount = sum(item.product.price * item.quantity for item in cart_items)
        transaction_time = datetime.now()
        time_24h_ago = transaction_time - timedelta(hours=24)
        orders_last_24h = Order.query.filter(
            Order.user_id == current_user.id,
            Order.timestamp >= time_24h_ago,
            Order.timestamp <= transaction_time,
        ).all()
        num_trans_24h = sum(1 for o in orders_last_24h if o.status == "Completed")
        num_failed_24h = sum(1 for o in orders_last_24h if o.status == "Cancelled")
        quantity = sum(item.quantity for item in cart_items)
        # Compute freq_last_24h: number of completed orders in last 24h
        freq_last_24h = Order.query.filter(
            Order.user_id == current_user.id,
            Order.timestamp >= time_24h_ago,
            Order.timestamp <= transaction_time,
            Order.status == "Completed"
        ).count()

        # Compute amount_last_24h: total amount spent in last 24h
        amount_last_24h = db.session.query(db.func.sum(Order.total_amount)).filter(
            Order.user_id == current_user.id,
            Order.timestamp >= time_24h_ago,
            Order.timestamp <= transaction_time,
            Order.status == "Completed"
        ).scalar() or 0.0

        # Compute sudden_category_switch: compare last completed order's category to current
        last_order = Order.query.filter(
            Order.user_id == current_user.id,
            Order.status == "Completed"
        ).order_by(Order.timestamp.desc()).first()
        if last_order:
            last_order_item = OrderItem.query.filter_by(order_id=last_order.id).first()
            if last_order_item and last_order_item.product:
                last_category = last_order_item.product.category
            else:
                last_category = None
            sudden_category_switch = int(last_category != category) if last_category else 0
        else:
            sudden_category_switch = 0
        sample_transaction = {
            "account_age_days": account_age_days,
            "payment_method": payment_method,
            "device": "Laptop",
            "category": category,
            "amount": total_amount,
            "quantity": quantity,
            "total_value": total_amount,
            "num_trans_24h": num_trans_24h,
            "num_failed_24h": num_failed_24h,
            "no_of_cards_from_ip": getattr(current_user, "no_of_cards_from_ip", 0),
            "promo_used": 0,  # or set based on your logic
            "freq_last_24h": freq_last_24h,
            "amount_last_24h": amount_last_24h,
            "sudden_category_switch": sudden_category_switch,
            "User_id": current_user.id,
            "TimeStamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        prediction = fraud_predictor.predict(sample_transaction)

        # Set order status based on prediction
        if prediction['decision'] == 'GENUINE':
            order_status = "Completed"
        elif prediction['decision'] == 'FRAUD':
            order_status = "Cancelled"
        elif prediction['decision'] == 'NEED TO TAKE FEEDBACK':
            order_status = "Needs Feedback"
        else:
            order_status = "Unknown"

        # Always create an Order record for every attempt
        order = Order(
            user_id=current_user.id,
            total_amount=total_amount,
            payment_method=payment_method,
            device="Laptop",
            status=order_status,
            timestamp=datetime.now(),
        )
        db.session.add(order)
        db.session.commit()

        if prediction['decision'] == 'GENUINE':
            # Payment is successful, create order items, clear cart, etc.
            for item in cart_items:
                order_item = OrderItem(
                    order_id=order.id, product_id=item.product.id, quantity=item.quantity
                )
                item.product.stock -= item.quantity
                db.session.add(order_item)
            CartItem.query.filter_by(user_id=current_user.id).delete()
            db.session.commit()
            return render_template('payment.html', payment_success=True, prediction=prediction, payment_method=payment_method, total_amount=total_amount)
        elif prediction['decision'] == 'NEED TO TAKE FEEDBACK':
            feedback_case = FeedbackCase(
                user_id=current_user.id,
                order_id=order.id,
                payment_method=payment_method,
                device="Laptop",
                category=category,
                amount=total_amount,
                quantity=quantity,
                total_value=total_amount,
                num_trans_24h=num_trans_24h,
                num_failed_24h=num_failed_24h,
                no_of_cards_from_ip=getattr(current_user, "no_of_cards_from_ip", 0),
                account_age_days=account_age_days,
                timestamp=datetime.now(),
                prediction=prediction['decision'],
                probability=prediction['probability'],
                anomaly_score=prediction['anomaly_score'],
                admin_status="Pending",
                product_id=product_id,  # <-- safe assignment
            )
            db.session.add(feedback_case)
            db.session.commit()
            return render_template('payment.html', payment_success=False, prediction=prediction, payment_method=payment_method, total_amount=total_amount, feedback_required=True)
        else:
            # Payment not successful, show model output
            return render_template('payment.html', payment_success=False, prediction=prediction, payment_method=payment_method, total_amount=total_amount)

    return render_template('payment.html')


@main_bp.route("/admin/feedback_cases")
@login_required
def admin_feedback_cases():
    if not current_user.is_admin:
        flash("Access denied.")
        return redirect(url_for("main_bp.index"))
    cases = FeedbackCase.query.order_by(FeedbackCase.timestamp.desc()).all()
    cases = [case for case in cases if case.product is not None]
    return render_template("admin_feedback_cases.html", cases=cases)


@main_bp.route("/admin/feedback_case/<int:case_id>/<action>")
@login_required
def feedback_case_action(case_id, action):
    if not current_user.is_admin:
        flash("Access denied.")
        return redirect(url_for("main_bp.index"))
    case = FeedbackCase.query.get_or_404(case_id)
    if action == "approve":
        case.admin_status = "Approved"
        # Update the related order to Completed
        if case.order_id:
            order = Order.query.get(case.order_id)
            if order:
                order.status = "Completed"
                # Create order items for all cart items at the time of payment
                # (Assuming you want to restore the cart items for this order)
                cart_items = CartItem.query.filter_by(user_id=case.user_id).all()
                for item in cart_items:
                    order_item = OrderItem(
                        order_id=order.id,
                        product_id=item.product_id,
                        quantity=item.quantity
                    )
                    product = Product.query.get(item.product_id)
                    if product:
                        product.stock -= item.quantity
                    db.session.add(order_item)
                # Clear the cart
                CartItem.query.filter_by(user_id=case.user_id).delete()
    elif action == "reject":
        case.admin_status = "Rejected"
        # Optionally, mark order as Cancelled
        if case.order_id:
            order = Order.query.get(case.order_id)
            if order:
                order.status = "Cancelled"
    db.session.commit()
    flash(f"Case {action}d.")
    return redirect(url_for("main_bp.admin_feedback_cases"))

from app import db
from app.models import FeedbackCase, Product

# Find all feedback cases with missing products
broken_cases = [case for case in FeedbackCase.query.all() if not Product.query.get(case.product_id)]
print(broken_cases)

# Optionally, delete them:
for case in broken_cases:
    db.session.delete(case)
db.session.commit()
