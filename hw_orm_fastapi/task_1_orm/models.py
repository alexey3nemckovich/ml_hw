from datetime import datetime
from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Publisher(Base):
    __tablename__ = "publishers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    books: Mapped[list["Book"]] = relationship("Book", cascade="all, delete")


class Book(Base):
    __tablename__ = "books"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    id_publisher: Mapped[int] = mapped_column(ForeignKey("publishers.id"), nullable=False)

    publisher: Mapped[Publisher] = relationship("Publisher", back_populates="books")
    stocks: Mapped[list["Stock"]] = relationship("Stock", cascade="all, delete")


class Shop(Base):
    __tablename__ = "shops"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    stocks: Mapped[list["Stock"]] = relationship("Stock", cascade="all, delete")


class Stock(Base):
    __tablename__ = "stocks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id_book: Mapped[int] = mapped_column(ForeignKey("books.id"), nullable=False)
    id_shop: Mapped[int] = mapped_column(ForeignKey("shops.id"), nullable=False)
    count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    book: Mapped[Book] = relationship("Book", back_populates="stocks")
    shop: Mapped[Shop] = relationship("Shop", back_populates="stocks")
    sales: Mapped[list["Sale"]] = relationship("Sale", cascade="all, delete")


class Sale(Base):
    __tablename__ = "sales"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    price: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    date_sale: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.now
    )
    id_stock: Mapped[int] = mapped_column(ForeignKey("stocks.id"), nullable=False)
    count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    stock: Mapped[Stock] = relationship("Stock", back_populates="sales")
