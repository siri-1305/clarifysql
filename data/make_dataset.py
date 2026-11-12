"""ClarifySQL dataset generator.
Creates ecommerce.db (SQLite) and questions.json (labelled test questions).
Run:  python make_dataset.py        (same output every time: fixed random seed)
"""
import json, random, sqlite3, os
from datetime import date, timedelta

random.seed(42)
DB, QF = "ecommerce.db", "questions.json"
if os.path.exists(DB):
    os.remove(DB)

# ---------------------------------------------------------------- reference data
CATEGORIES = ["Electronics", "Fashion", "Home & Kitchen", "Books", "Sports", "Beauty"]
PRODUCTS = {  # name: (category, price INR)
    "Wireless Earbuds": ("Electronics", 2499), "Smart Watch": ("Electronics", 4999),
    "Bluetooth Speaker": ("Electronics", 1799), "Power Bank 20000mAh": ("Electronics", 1599),
    "Gaming Mouse": ("Electronics", 1299), "Mechanical Keyboard": ("Electronics", 3499),
    "4K Action Camera": ("Electronics", 8999), "USB-C Hub": ("Electronics", 999),
    "Cotton T-Shirt": ("Fashion", 599), "Slim Fit Jeans": ("Fashion", 1499),
    "Running Shoes": ("Fashion", 2999), "Leather Wallet": ("Fashion", 799),
    "Denim Jacket": ("Fashion", 2499), "Silk Saree": ("Fashion", 5999), "Sports Cap": ("Fashion", 299),
    "Non-stick Pan Set": ("Home & Kitchen", 1899), "Electric Kettle": ("Home & Kitchen", 1099),
    "Air Fryer": ("Home & Kitchen", 6499), "Memory Foam Pillow": ("Home & Kitchen", 1199),
    "LED Desk Lamp": ("Home & Kitchen", 899), "Vacuum Flask": ("Home & Kitchen", 699),
    "Python Programming Guide": ("Books", 499), "Data Structures Handbook": ("Books", 649),
    "Indian History Essentials": ("Books", 399), "Startup Playbook": ("Books", 549),
    "Mindfulness Journal": ("Books", 299),
    "Yoga Mat": ("Sports", 799), "Dumbbell Set 10kg": ("Sports", 1999), "Cricket Bat": ("Sports", 2299),
    "Badminton Racket": ("Sports", 1399), "Resistance Bands": ("Sports", 449),
    "Vitamin C Serum": ("Beauty", 699), "Aloe Vera Gel": ("Beauty", 249), "Matte Lipstick": ("Beauty", 499),
    "Hair Oil": ("Beauty", 349), "Sunscreen SPF 50": ("Beauty", 429),
}
CITIES = [("Hyderabad", "Telangana"), ("Visakhapatnam", "Andhra Pradesh"), ("Vijayawada", "Andhra Pradesh"),
          ("Srikakulam", "Andhra Pradesh"), ("Bengaluru", "Karnataka"), ("Chennai", "Tamil Nadu"),
          ("Mumbai", "Maharashtra"), ("Pune", "Maharashtra"), ("Delhi", "Delhi"), ("Kolkata", "West Bengal")]
FIRST = ["Aarav", "Vivaan", "Aditya", "Sai", "Arjun", "Rohan", "Karthik", "Ravi", "Suresh", "Anil", "Priya",
         "Ananya", "Sneha", "Divya", "Lakshmi", "Pooja", "Kavya", "Meera", "Harsha", "Sirisha"]
LAST = ["Reddy", "Naidu", "Sharma", "Kumar", "Rao", "Patel", "Singh", "Gupta", "Iyer", "Das", "Varma", "Chowdary"]
PAY = [("UPI", 40), ("Credit Card", 20), ("Debit Card", 15), ("Cash on Delivery", 15), ("Net Banking", 10)]
STATUS = [("delivered", 70), ("shipped", 10), ("cancelled", 12), ("returned", 8)]
COMMENTS = {5: ["Excellent quality", "Loved it", "Worth every rupee"], 4: ["Good product", "Happy with it", "Nice value"],
            3: ["Average", "Okay for the price", "Does the job"], 2: ["Not great", "Below expectations"],
            1: ["Poor quality", "Waste of money"]}
START, END = date(2023, 1, 1), date(2024, 12, 31)

def wchoice(pairs):
    return random.choices([p[0] for p in pairs], weights=[p[1] for p in pairs])[0]

# ---------------------------------------------------------------- schema
con = sqlite3.connect(DB)
con.executescript("""
CREATE TABLE categories (category_id INTEGER PRIMARY KEY, category_name TEXT NOT NULL);
CREATE TABLE products (product_id INTEGER PRIMARY KEY, product_name TEXT NOT NULL,
  category_id INTEGER REFERENCES categories(category_id), price REAL NOT NULL, cost REAL NOT NULL,
  stock_qty INTEGER NOT NULL, launch_date TEXT);
CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, full_name TEXT NOT NULL, email TEXT,
  city TEXT, state TEXT, signup_date TEXT);
CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(customer_id),
  order_date TEXT NOT NULL, status TEXT NOT NULL, payment_method TEXT);
CREATE TABLE order_items (order_item_id INTEGER PRIMARY KEY, order_id INTEGER REFERENCES orders(order_id),
  product_id INTEGER REFERENCES products(product_id), quantity INTEGER NOT NULL, unit_price REAL NOT NULL);
CREATE TABLE reviews (review_id INTEGER PRIMARY KEY, product_id INTEGER REFERENCES products(product_id),
  customer_id INTEGER REFERENCES customers(customer_id), rating INTEGER NOT NULL, review_date TEXT, comment TEXT);
""")

for i, c in enumerate(CATEGORIES, 1):
    con.execute("INSERT INTO categories VALUES (?,?)", (i, c))
cat_id = {c: i for i, c in enumerate(CATEGORIES, 1)}

# products: popularity, quality and margin are independent, so "best" really is ambiguous
pop, qual = {}, {}
stock_zero = random.sample(list(PRODUCTS), 3)
stock_low = random.sample([p for p in PRODUCTS if p not in stock_zero], 5)
for pid, (name, (cat, price)) in enumerate(PRODUCTS.items(), 1):
    margin = random.uniform(0.15, 0.55)
    stock = 0 if name in stock_zero else random.randint(2, 18) if name in stock_low else random.randint(20, 150)
    launch = date(2022, 6, 1) + timedelta(days=random.randint(0, 900))
    con.execute("INSERT INTO products VALUES (?,?,?,?,?,?,?)",
                (pid, name, cat_id[cat], price, round(price * (1 - margin), 2), stock, launch.isoformat()))
    pop[pid], qual[pid] = random.uniform(0.3, 3.0), random.uniform(2.8, 4.8)

# customers (about 10% never order)
cust = []
for cid in range(1, 201):
    city, state = random.choice(CITIES)
    name = f"{random.choice(FIRST)} {random.choice(LAST)}"
    signup = date(2022, 6, 1) + timedelta(days=random.randint(0, 850))
    con.execute("INSERT INTO customers VALUES (?,?,?,?,?,?)",
                (cid, name, f"{name.lower().replace(' ', '.')}{cid}@example.com", city, state, signup.isoformat()))
    cust.append((cid, signup))
inactive = set(random.sample(range(1, 201), 20))
cweight = {c: random.lognormvariate(0, 0.9) for c, _ in cust if c not in inactive}

# orders and items
pids = list(PRODUCTS.values()) and list(range(1, len(PRODUCTS) + 1))
price_of = {pid: p for pid, (_, p) in enumerate(PRODUCTS.values(), 1)}
delivered_items = []
oi = 1
for oid in range(1, 1301):
    c = random.choices(list(cweight), weights=list(cweight.values()))[0]
    signup = dict(cust)[c]
    lo = max(START, signup)
    od = lo + timedelta(days=random.randint(0, (END - lo).days))
    status = wchoice(STATUS)
    con.execute("INSERT INTO orders VALUES (?,?,?,?,?)", (oid, c, od.isoformat(), status, wchoice(PAY)))
    for p in random.choices(pids, weights=[pop[x] for x in pids], k=random.randint(1, 4)):
        con.execute("INSERT INTO order_items VALUES (?,?,?,?,?)", (oi, oid, p, random.randint(1, 3), price_of[p]))
        if status == "delivered":
            delivered_items.append((c, p, od))
        oi += 1

# reviews: about 35% of delivered items, one review per customer-product pair
seen, rid = set(), 1
for c, p, od in delivered_items:
    if random.random() < 0.35 and (c, p) not in seen:
        seen.add((c, p))
        rating = max(1, min(5, round(random.gauss(qual[p], 0.9))))
        rd = min(END, od + timedelta(days=random.randint(3, 20)))
        con.execute("INSERT INTO reviews VALUES (?,?,?,?,?,?)",
                    (rid, p, c, rating, rd.isoformat(), random.choice(COMMENTS[rating])))
        rid += 1
con.commit()

# ---------------------------------------------------------------- questions
REV = "SUM(oi.quantity * oi.unit_price)"
DEL = "JOIN orders o ON o.order_id = oi.order_id WHERE o.status = 'delivered'"
clear = [
 ("How many customers are there?", "SELECT COUNT(*) FROM customers;"),
 ("How many orders were cancelled?", "SELECT COUNT(*) FROM orders WHERE status = 'cancelled';"),
 ("List all product names in the Electronics category.", "SELECT p.product_name FROM products p JOIN categories c ON p.category_id = c.category_id WHERE c.category_name = 'Electronics';"),
 ("What is the price of the Wireless Earbuds?", "SELECT price FROM products WHERE product_name = 'Wireless Earbuds';"),
 ("How many orders were placed in 2024?", "SELECT COUNT(*) FROM orders WHERE order_date >= '2024-01-01' AND order_date < '2025-01-01';"),
 ("How many customers are from Hyderabad?", "SELECT COUNT(*) FROM customers WHERE city = 'Hyderabad';"),
 ("Show the 5 most expensive products.", "SELECT product_name, price FROM products ORDER BY price DESC LIMIT 5;"),
 ("What is the average rating of the Wireless Earbuds?", "SELECT AVG(r.rating) FROM reviews r JOIN products p ON p.product_id = r.product_id WHERE p.product_name = 'Wireless Earbuds';"),
 ("How many products are out of stock?", "SELECT COUNT(*) FROM products WHERE stock_qty = 0;"),
 ("What is the total quantity sold of the Yoga Mat?", "SELECT SUM(oi.quantity) FROM order_items oi JOIN products p ON p.product_id = oi.product_id WHERE p.product_name = 'Yoga Mat';"),
 ("How many customers signed up in 2023?", "SELECT COUNT(*) FROM customers WHERE signup_date >= '2023-01-01' AND signup_date < '2024-01-01';"),
 ("How many orders were paid by UPI?", "SELECT COUNT(*) FROM orders WHERE payment_method = 'UPI';"),
 ("Which category has the most products?", "SELECT c.category_name FROM categories c JOIN products p ON p.category_id = c.category_id GROUP BY c.category_id ORDER BY COUNT(*) DESC LIMIT 1;"),
 ("What is the total revenue from delivered orders in 2024?", f"SELECT {REV} FROM order_items oi {DEL} AND o.order_date >= '2024-01-01';"),
 ("How many reviews have a rating of 5?", "SELECT COUNT(*) FROM reviews WHERE rating = 5;"),
 ("How many customers from Visakhapatnam placed more than 3 orders?", "SELECT COUNT(*) FROM (SELECT c.customer_id FROM customers c JOIN orders o ON o.customer_id = c.customer_id WHERE c.city = 'Visakhapatnam' GROUP BY c.customer_id HAVING COUNT(*) > 3);"),
 ("What is the average order value for delivered orders?", f"SELECT AVG(t) FROM (SELECT {REV} AS t FROM order_items oi {DEL} GROUP BY o.order_id);"),
 ("List the products launched after 2024-01-01.", "SELECT product_name FROM products WHERE launch_date > '2024-01-01';"),
]
def top(expr, joins, grp, lim=5, where=""):
    return f"SELECT {grp}, {expr} AS score FROM {joins} {where} GROUP BY {grp} ORDER BY score DESC LIMIT {lim};"
PJ = "products p JOIN order_items oi ON oi.product_id = p.product_id"
vague = [
 ("Show me the best products.", "Best by what?", [("highest sales (units sold)", top("SUM(oi.quantity)", PJ, "p.product_name")),
   ("highest average rating", top("AVG(r.rating)", "products p JOIN reviews r ON r.product_id = p.product_id", "p.product_name")),
   ("highest profit", top("SUM(oi.quantity * (oi.unit_price - p.cost))", PJ, "p.product_name"))]),
 ("Who are our top customers?", "Top by what?", [("total spend", top(REV, "customers c JOIN orders o ON o.customer_id = c.customer_id JOIN order_items oi ON oi.order_id = o.order_id", "c.full_name")),
   ("number of orders", top("COUNT(DISTINCT o.order_id)", "customers c JOIN orders o ON o.customer_id = c.customer_id", "c.full_name"))]),
 ("Show recent orders.", "How recent?", [("the 10 latest orders", "SELECT * FROM orders ORDER BY order_date DESC LIMIT 10;"),
   ("orders in the last 30 days of data", "SELECT * FROM orders WHERE order_date >= '2024-12-02';")]),
 ("Which products are popular?", "Popular in what sense?", [("most units sold", top("SUM(oi.quantity)", PJ, "p.product_name")),
   ("appear in the most orders", top("COUNT(DISTINCT oi.order_id)", PJ, "p.product_name"))]),
 ("Show sales by region.", "Which region level?", [("by city", f"SELECT c.city, {REV} FROM customers c JOIN orders o ON o.customer_id = c.customer_id JOIN order_items oi ON oi.order_id = o.order_id GROUP BY c.city;"),
   ("by state", f"SELECT c.state, {REV} FROM customers c JOIN orders o ON o.customer_id = c.customer_id JOIN order_items oi ON oi.order_id = o.order_id GROUP BY c.state;")]),
 ("What is our revenue?", "Which orders count?", [("all orders", f"SELECT {REV} FROM order_items oi;"),
   ("delivered orders only", f"SELECT {REV} FROM order_items oi {DEL};")]),
 ("List expensive products.", "How expensive?", [("the 5 highest-priced", "SELECT product_name, price FROM products ORDER BY price DESC LIMIT 5;"),
   ("priced above the average price", "SELECT product_name, price FROM products WHERE price > (SELECT AVG(price) FROM products);")]),
 ("Which category performs best?", "Best by what?", [("revenue", top(REV, "categories c JOIN products p ON p.category_id = c.category_id JOIN order_items oi ON oi.product_id = p.product_id", "c.category_name", 1)),
   ("average rating", top("AVG(r.rating)", "categories c JOIN products p ON p.category_id = c.category_id JOIN reviews r ON r.product_id = p.product_id", "c.category_name", 1))]),
 ("Show active customers.", "Active how?", [("ordered in 2024", "SELECT DISTINCT c.full_name FROM customers c JOIN orders o ON o.customer_id = c.customer_id WHERE o.order_date >= '2024-01-01';"),
   ("ordered in the last 6 months of data", "SELECT DISTINCT c.full_name FROM customers c JOIN orders o ON o.customer_id = c.customer_id WHERE o.order_date >= '2024-07-01';")]),
 ("Find low stock products.", "What counts as low?", [("stock below 10", "SELECT product_name, stock_qty FROM products WHERE stock_qty < 10;"),
   ("stock below 20", "SELECT product_name, stock_qty FROM products WHERE stock_qty < 20;")]),
 ("Show good reviews.", "What counts as good?", [("rating 4 or 5", "SELECT * FROM reviews WHERE rating >= 4;"), ("rating 5 only", "SELECT * FROM reviews WHERE rating = 5;")]),
 ("How are sales this year?", "Total or monthly?", [("total 2024 revenue", f"SELECT {REV} FROM order_items oi {DEL} AND o.order_date >= '2024-01-01';"),
   ("month-wise 2024 revenue", f"SELECT substr(o.order_date,1,7) AS month, {REV} FROM order_items oi {DEL} AND o.order_date >= '2024-01-01' GROUP BY month ORDER BY month;")]),
 ("Who bought the most?", "Most by what?", [("quantity of items", top("SUM(oi.quantity)", "customers c JOIN orders o ON o.customer_id = c.customer_id JOIN order_items oi ON oi.order_id = o.order_id", "c.full_name", 1)),
   ("total spend", top(REV, "customers c JOIN orders o ON o.customer_id = c.customer_id JOIN order_items oi ON oi.order_id = o.order_id", "c.full_name", 1))]),
 ("Which is the best selling month?", "Best by what?", [("revenue", top(REV, "orders o JOIN order_items oi ON oi.order_id = o.order_id", "substr(o.order_date,1,7)", 1)),
   ("number of orders", top("COUNT(*)", "orders o", "substr(o.order_date,1,7)", 1))]),
 ("Which customers are inactive?", "Inactive how?", [("never placed an order", "SELECT full_name FROM customers WHERE customer_id NOT IN (SELECT customer_id FROM orders);"),
   ("no order in 2024", "SELECT full_name FROM customers WHERE customer_id NOT IN (SELECT customer_id FROM orders WHERE order_date >= '2024-01-01');")]),
]
tricky = [  # (question, sql or None, note)
 ("How many orders were completed?", "SELECT COUNT(*) FROM orders WHERE status = 'delivered';", "'completed' is stored as 'delivered'"),
 ("List customers from Bangalore.", "SELECT full_name FROM customers WHERE city = 'Bengaluru';", "city stored as 'Bengaluru'"),
 ("How many orders were paid by cash?", "SELECT COUNT(*) FROM orders WHERE payment_method = 'Cash on Delivery';", "value is 'Cash on Delivery'"),
 ("Show orders that were refunded.", "SELECT * FROM orders WHERE status = 'returned';", "'refunded' maps to status 'returned'"),
 ("What is the total amount of order 10?", "SELECT SUM(quantity * unit_price) FROM order_items WHERE order_id = 10;", "no amount column; compute from items"),
 ("Show products in the Electronic category.", "SELECT p.product_name FROM products p JOIN categories c ON p.category_id = c.category_id WHERE c.category_name = 'Electronics';", "'Electronic' vs 'Electronics'"),
 ("How many users signed up in 2023?", "SELECT COUNT(*) FROM customers WHERE signup_date >= '2023-01-01' AND signup_date < '2024-01-01';", "'users' means customers table"),
 ("What is the total quantity of items ordered by customers in Hyderabad?", "SELECT SUM(oi.quantity) FROM order_items oi JOIN orders o ON o.order_id = oi.order_id JOIN customers c ON c.customer_id = o.customer_id WHERE c.city = 'Hyderabad';", "three-table join"),
 ("List item names with price above 5000.", "SELECT product_name FROM products WHERE price > 5000;", "'item' means product"),
 ("How many orders were made with a debit card?", "SELECT COUNT(*) FROM orders WHERE payment_method = 'Debit Card';", "value is 'Debit Card'"),
 ("Show the product with the highest margin.", "SELECT product_name, price - cost AS margin FROM products ORDER BY margin DESC LIMIT 1;", "no margin column; compute price - cost"),
 ("Show customer phone numbers.", None, "UNANSWERABLE: no phone column"),
 ("Show employee salaries.", None, "UNANSWERABLE: no employee data"),
]
qs, n = [], 1
for q, sql in clear:
    qs.append({"id": f"C{n:02d}", "type": "clear", "question": q, "should_ask": False, "expected_sql": sql}); n += 1
n = 1
for q, ask, opts in vague:
    qs.append({"id": f"V{n:02d}", "type": "vague", "question": q, "should_ask": True, "clarifying_question": ask,
               "interpretations": [{"label": l, "expected_sql": s} for l, s in opts]}); n += 1
n = 1
for q, sql, note in tricky:
    qs.append({"id": f"T{n:02d}", "type": "tricky", "question": q, "should_ask": False,
               "answerable": sql is not None, "expected_sql": sql, "note": note}); n += 1

# ---------------------------------------------------------------- verify every SQL runs
bad = 0
for q in qs:
    for sql in ([q["expected_sql"]] if "expected_sql" in q else [i["expected_sql"] for i in q["interpretations"]]):
        if sql is None:
            continue
        try:
            rows = con.execute(sql).fetchall()
            if not rows or rows[0][0] in (None, 0) and len(rows) == 1:
                print("WARN empty/zero result:", q["id"], q["question"]); bad += 1
        except Exception as e:
            print("ERROR", q["id"], e); bad += 1
json.dump(qs, open(QF, "w"), indent=2, ensure_ascii=False)
for t in ["categories", "products", "customers", "orders", "order_items", "reviews"]:
    print(f"{t:12s}", con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0], "rows")
print(len(qs), "questions |", bad, "problems")
