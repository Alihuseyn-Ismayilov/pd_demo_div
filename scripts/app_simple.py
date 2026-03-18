import streamlit as st

# ─────────────────────────────────────────────
# STEP 1: Give the page a title
# ─────────────────────────────────────────────
st.title("Simple Calculator")
st.write("This is a Streamlit app. No ML, no styling. Just inputs and a button.")

# ─────────────────────────────────────────────
# STEP 2: Create input fields
# ─────────────────────────────────────────────
number_1 = st.number_input("Number 1", value=0.0)
number_2 = st.number_input("Number 2", value=0.0)

# ─────────────────────────────────────────────
# STEP 3: Let the user pick an operation
# ─────────────────────────────────────────────
operation = st.selectbox(
    "Choose an operation",
    ["Add", "Subtract", "Multiply", "Divide"]
)

# ─────────────────────────────────────────────
# STEP 4: Button — nothing runs until clicked
# ─────────────────────────────────────────────
if st.button("Calculate"):

    if operation == "Add":
        result = number_1 + number_2
        st.success(f"Result:  {number_1}  +  {number_2}  =  {result}")

    elif operation == "Subtract":
        result = number_1 - number_2
        st.success(f"Result:  {number_1}  -  {number_2}  =  {result}")

    elif operation == "Multiply":
        result = number_1 * number_2
        st.success(f"Result:  {number_1}  ×  {number_2}  =  {result}")

    elif operation == "Divide":
        if number_2 == 0:
            st.error("Cannot divide by zero.")
        else:
            result = number_1 / number_2
            st.success(f"Result:  {number_1}  ÷  {number_2}  =  {result:.4f}")
