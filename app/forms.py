"""
forms.py

Defines all Flask-WTF forms used across the eCommerce web app.
Includes forms for user auth, product management, checkout, cart updates, profile editing, and reviews.
"""

from flask_wtf import FlaskForm
from wtforms import (
    StringField, PasswordField, SubmitField,
    BooleanField, FloatField, IntegerField, TextAreaField,
    SelectField
)
from wtforms.validators import (
    DataRequired, Email, EqualTo, Length,
    NumberRange, URL
)


class RegisterForm(FlaskForm):
    """Form for new user registration."""
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField("Confirm Password", validators=[DataRequired(), EqualTo("password")])
    submit = SubmitField("Register")


class LoginForm(FlaskForm):
    """Form for existing user login."""
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    remember = BooleanField("Remember Me")
    submit = SubmitField("Login")


class ProductForm(FlaskForm):
    """Form used by admins to add/edit products."""
    name = StringField("Product Name", validators=[DataRequired()])
    description = TextAreaField("Description")
    price = FloatField("Price", validators=[DataRequired(), NumberRange(min=0)])
    image_url = StringField("Image URL", validators=[URL()])
    category = StringField("Category")
    stock = IntegerField("Stock", validators=[DataRequired(), NumberRange(min=0)])
    submit = SubmitField("Save Product")


class QuantityForm(FlaskForm):
    """Form for updating quantity in the cart."""
    quantity = IntegerField("Quantity", validators=[DataRequired(), NumberRange(min=1)])
    submit = SubmitField("Update")


class CheckoutForm(FlaskForm):
    """Form to confirm checkout process."""
    submit = SubmitField("Proceed to Checkout")


class ProfileForm(FlaskForm):
    """Form for editing user address and shopping preferences."""
    address = StringField("Shipping Address", validators=[Length(max=300)])
    preferences = TextAreaField("Shopping Preferences", validators=[Length(max=1000)])
    submit = SubmitField("Update Profile")


class ReviewForm(FlaskForm):
    """Form for submitting product reviews."""
    rating = SelectField(
        "Rating",
        choices=[(str(i), f"{i} Stars") for i in range(1, 6)],
        validators=[DataRequired()]
    )
    content = TextAreaField("Comment", validators=[DataRequired(), Length(max=1000)])
    submit = SubmitField("Submit Review")
