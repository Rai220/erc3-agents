import time
import json
from typing import Annotated, List, Union, Literal, Optional
from pydantic import BaseModel, Field
from erc3 import store, ApiException, TaskInfo, ERC3
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

def fetch_all_products(store_api):
    all_products = []
    offset = 0
    limit = 3 # Limit set to 3 to comply with potential server restrictions
    while True:
        req = store.Req_ListProducts(offset=offset, limit=limit)
        try:
            res = store_api.dispatch(req)
            if hasattr(res, 'products') and res.products:
                all_products.extend([p.model_dump(exclude_none=True) for p in res.products])
            
            if res.next_offset == -1:
                break
            offset = res.next_offset
        except ApiException as e:
            print(f"Warning: Error fetching products batch: {e}")
            break
    return all_products

def get_tools(store_api):
    """Create tools bound to the specific store_api instance."""
    
    # list_products tool removed as products are now pre-fetched
    
    @tool
    def view_basket():
        """View current shopping basket.
        Returns items in basket, subtotal, discount, and total price."""
        req = store.Req_ViewBasket()
        try:
            res = store_api.dispatch(req)
            return res.model_dump(exclude_none=True)
        except ApiException as e:
            return f"Error: {e.detail}"

    @tool
    def add_product(sku: str, quantity: int = 1):
        """Add a product to the shopping basket by SKU.
        Returns the updated basket status."""
        req = store.Req_AddProductToBasket(sku=sku, quantity=quantity)
        try:
            res = store_api.dispatch(req)
            return res.model_dump(exclude_none=True)
        except ApiException as e:
            return f"Error: {e.detail}"

    @tool
    def remove_product(sku: str, quantity: int = 1):
        """Remove a product from the shopping basket by SKU.
        Returns the updated basket status."""
        req = store.Req_RemoveItemFromBasket(sku=sku, quantity=quantity)
        try:
            res = store_api.dispatch(req)
            return res.model_dump(exclude_none=True)
        except ApiException as e:
            return f"Error: {e.detail}"

    @tool
    def checkout():
        """Checkout and complete the purchase.
        This finalizes the transaction. Use this when you have added all necessary items."""
        req = store.Req_CheckoutBasket()
        try:
            res = store_api.dispatch(req)
            return res.model_dump(exclude_none=True)
        except ApiException as e:
            return f"Error: {e.detail}"

    @tool
    def apply_coupon(coupon: str):
        """Apply a coupon code to the basket.
        Use this to get discounts."""
        req = store.Req_ApplyCoupon(coupon=coupon)
        try:
            res = store_api.dispatch(req)
            return res.model_dump(exclude_none=True)
        except ApiException as e:
            return f"Error: {e.detail}"

    @tool
    def remove_coupon():
        """Remove the currently applied coupon from the basket."""
        req = store.Req_RemoveCoupon()
        try:
            res = store_api.dispatch(req)
            return res.model_dump(exclude_none=True)
        except ApiException as e:
            return f"Error: {e.detail}"

    @tool
    def find_best_coupon(coupons: List[str]):
        """Try multiple coupons on the CURRENT basket content and apply the best one.
        IMPORTANT: This does NOT simulate adding/removing items. It only tests coupons on what is currently in the basket.
        Args:
            coupons: List of coupon codes to test.
        Returns:
            Report of tested coupons and the final applied best coupon.
        """
        if not coupons:
            return "No coupons provided."
        
        best_coupon = None
        
        # First remove any existing coupon to start fresh
        try:
            store_api.dispatch(store.Req_RemoveCoupon())
        except:
            pass 

        try:
            initial_basket = store_api.dispatch(store.Req_ViewBasket())
            base_total = initial_basket.total
            best_total = base_total
            report = [f"No coupon: {base_total}"]
            
            for code in coupons:
                try:
                    store_api.dispatch(store.Req_ApplyCoupon(coupon=code))
                    basket = store_api.dispatch(store.Req_ViewBasket())
                    total = basket.total
                    report.append(f"{code}: {total}")
                    
                    if total < best_total:
                        best_total = total
                        best_coupon = code
                except ApiException as e:
                    report.append(f"{code}: Invalid ({e.detail})")
                except Exception:
                    report.append(f"{code}: Error")

            # Apply best coupon
            if best_coupon:
                try:
                    store_api.dispatch(store.Req_ApplyCoupon(coupon=best_coupon))
                    return f"Tested: {'; '.join(report)}. Applied best: {best_coupon} (Total: {best_total})"
                except Exception as e:
                    return f"Error re-applying best coupon {best_coupon}: {e}"
            else:
                # Ensure no coupon is applied
                try:
                    store_api.dispatch(store.Req_RemoveCoupon())
                except:
                    pass
                return f"Tested: {'; '.join(report)}. No coupon applied (Base price {base_total} was best)."
        except ApiException as e:
             return f"Error accessing basket: {e.detail}"

    return [view_basket, add_product, remove_product, checkout, apply_coupon, remove_coupon, find_best_coupon]

def run_agent(model: str, api: ERC3, task: TaskInfo):
    store_api = api.get_store_client(task)
    
    # Pre-fetch all products
    print("Fetching all products...")
    products = fetch_all_products(store_api)
    products_str = json.dumps(products, indent=2)
    print(f"Fetched {len(products)} products.")

    tools = get_tools(store_api)
    
    # Use the passed model, or default to gpt-4o
    llm_model = model or "gpt-4o"
    
    print(f"Init agent with {llm_model} for task: {task.task_text}")

    llm = ChatOpenAI(model=llm_model, temperature=0)
    # Bind tools with parallel_tool_calls=False to prevent multiple tool calls in one step
    llm = llm.bind_tools(tools, parallel_tool_calls=False)

    system_prompt = f"""You are an expert shopping agent. Goal: 100% score.

HERE IS THE LIST OF AVAILABLE PRODUCTS:
{products_str}

EXAMPLE (ZERO-SHOT):
Task: "Buy 24 sodas as cheap as possible. Coupons: SALEX (when buying a lot of 6pk), BULK24 (for 24pk), COMBO (when buying 6pk and 12pk)"
Reasoning:
- The agent explored all combinations to reach quantity 24 (e.g. 4x6pk, 1x24pk, 2x12pk, 2x6pk+1x12pk).
- It found that buying 2x soda-6pk ($12 ea) and 1x soda-12pk ($20 ea) gave a subtotal of $44.
- Applying coupon SALEX to this specific combo gave a $14 discount, resulting in a total of $30.
- This was cheaper than other options (like 1x24pk which might be $40 with coupon).
Correct Result: *Evt_BasketCheckedOut{{Items:[{{soda-6pk 2 12}} {{soda-12pk 1 20}}], Subtotal:44, Coupon:SALEX, Discount:14, Total:30}}

RULES:
1. **Product Search**: You already have the full list of products above. DO NOT try to search or list products again. Use the list provided.
2. **Precise Matching**:
   - Verify EVERY adjective (Color, Model, Specs). "iPhone 15 Pro" != "iPhone 15 Pro Max".
   - If looking for "Blue", do NOT buy "Black".
3. **Optimization & Math (CRITICAL)**:
   - **Divide and Conquer Strategy**: When minimizing cost for a fixed quantity (e.g., "24 sodas"), you MUST test valid combinations SEQUENTIALLY.
   - **Step 1: List Candidates**: Identify different ways to reach the exact quantity (e.g., 1x24pk, 2x12pk, 4x6pk, 2x6pk+1x12pk).
   - **Step 2: Sequential Testing**:
     - **NEVER** add all options to the basket at once. `find_best_coupon` ONLY checks the *current* basket.
     - Loop through each candidate:
       1. ENSURE BASKET IS EMPTY (use `remove_product` or start fresh).
       2. Add the specific items for *this one candidate*.
       3. Use `find_best_coupon` to see the lowest price for this configuration.
       4. Record the price.
       5. Remove items (to prepare for the next candidate).
   - **Step 3: Select & Execute**: Compare the recorded prices. Re-build the basket with the Single Cheapest Combination and checkout.
   - **Coupons**: Gather ALL potential coupon codes found in the task or product list and test them on EACH candidate basket.
4. **Efficiency**:
   - **Batching**: You can add multiple items for a SINGLE candidate combination in sequence. Do NOT batch items from different solution candidates.
   - **Verification**: You MUST call `view_basket` immediately after EVERY `apply_coupon` (or `find_best_coupon`) to verify if it worked and check the new price.
   - **Final Check**: `view_basket` ONCE more before `checkout`.
   - **Pre-Checkout Reasoning (MANDATORY)**:
     - You are NOT allowed to checkout blindly.
     - Before calling `checkout`, output a reasoning block:
       1. **Recap**: List all valid combinations tested and their final prices.
       2. **Verification**: Confirm the current basket matches the BEST (cheapest) combination.
       3. **Decision**: State "Price $X is the lowest. Proceeding to checkout."
5. **Strict Constraints (CRITICAL)**:
   - **ALL CONDITIONS MUST BE MET**: If the task requires specific items, specific quantities, specific coupons, or staying under a specific budget, and ANY of these cannot be fully satisfied, you MUST NOT CHECKOUT.
   - **Coupon Failures**: If the task says "using coupon X" and coupon X is invalid or doesn't apply -> STOP. DO NOT CHECKOUT.
   - **Impossible Tasks**: If the task is impossible for ANY reason (stock, budget, coupon, item existence) -> STOP. CLEAR THE BASKET. FINISH WITHOUT CHECKOUT.
   - **Better Safe Than Sorry**: It is better to finish without buying than to buy the wrong thing or ignore a constraint. Finishing without purchase when constraints fail is the CORRECT behavior.

6. **Handling Checkout Errors (Dynamic Inventory)**:
   - If `checkout` fails with "insufficient inventory" (e.g., "available X, in basket Y"):
     - This is a RACE CONDITION (someone bought it before you).
     - **ACTION**: Calculate difference (Y - X). Remove that amount from basket using `remove_product`.
     - **RETRY**: Call `checkout` again immediately.
   - **NEVER give up** on "insufficient inventory" errors during checkout. Adjust and retry.

7. **Final Execution**:
   - **DEFAULT**: If ALL constraints are met and you have the best price -> CALL `checkout`.
   - **PROHIBITION**: IF ANY CONSTRAINT FAILED (missing item, budget exceeded, coupon failed, etc.) -> DO NOT CALL `checkout`. Just stop.
   - **Do NOT ask for user confirmation**.
   - **Do NOT say "ready to checkout"**.
"""

    graph = create_react_agent(llm, tools, prompt=system_prompt)
    
    inputs = {"messages": [HumanMessage(content=task.task_text)]}
    
    # Run the agent
    # We use a recursion limit to prevent infinite loops, but allow enough for pagination
    try:
        printed_count = 0
        for step in graph.stream(inputs, config={"recursion_limit": 50}, stream_mode="values"):
             messages = step["messages"]
             for message in messages[printed_count:]:
                 if message.type == "tool":
                     print(f"\n[Function Result] {message.name}: {message.content}")
                 elif message.content:
                     print(f"\n[{message.type}]: {message.content}")
                 if hasattr(message, "tool_calls") and message.tool_calls:
                     for tc in message.tool_calls:
                         print(f"\n[Tool Call]: {tc['name']} params={tc['args']}")
             printed_count = len(messages)
    
    except Exception as e:
        print(f"Agent failed with error: {e}")
