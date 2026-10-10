**Bitcoin Prediction Model**

## Main Ideas

- Target: Price change (in percentage) each hour (range -1 -> whatever)
- Features (hourly):
    - Volume Traded
    - Open Price
    - Close Price
    - Low Price
    - High Price
    - *More datasets*
- Input: 7 days tensor (7 days, 24 Hours, 6 Elements)
- Output: 1 day array of hourly target (24 hour elements)
---
- Test + train period = 8 days (7 test + 1 train)
