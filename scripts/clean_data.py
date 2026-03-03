"""Clean Rossmann data: load raw CSVs → clean → enrich → save."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.cleaner import (
    add_competition_days,
    add_holiday_proximity,
    add_log_sales,
    add_promo_month,
    add_store_stats,
    fill_missing,
    fix_types,
    load_raw,
    merge_store,
    remove_outliers,
    save,
    validate,
)


def main():
    parser = argparse.ArgumentParser(description="Clean Rossmann data")
    parser.add_argument("--raw-dir", default="data/raw", help="Raw data directory")
    parser.add_argument("--out-dir", default="data/cleaned", help="Output directory")
    parser.add_argument("--format", default="csv", choices=["feather", "csv"])
    parser.add_argument("--skip-test", action="store_true", help="Skip test set cleaning")
    parser.add_argument("--verbose", action="store_true", help="Print quality report")
    args = parser.parse_args()

    print("Loading raw data...")
    train, store, test = load_raw(args.raw_dir)
    print(f"  train: {len(train):,} rows | store: {len(store):,} rows | test: {len(test):,} rows")

    # Clean store data (shared between train and test)
    store = fill_missing(store)

    # --- Train pipeline ---
    print("Cleaning train...")
    # train chỉ có sales, chưa có thông tin cửa hàng -> ghép bảng store vào để có metadata
    train = merge_store(train, store)
    # sau merge nhiều cột bị NaN (CompetitionDistance, Promo2...) -> điền giá trị thay thế
    train = fill_missing(train)
    # cột Date đang string, StateHoliday đang số -> ép đúng kiểu datetime, category, int
    train = fix_types(train)
    # một số dòng có Sales bất thường (quá cao/thấp) -> loại bỏ để tránh lệch model
    train = remove_outliers(train)
    # model cần biết ngày lễ gần/xa thế nào -> thêm cột số ngày đến kỳ nghỉ gần nhất
    train = add_holiday_proximity(train)
    # mỗi cửa hàng có mức sales khác nhau -> thêm mean/std sales theo từng store
    train = add_store_stats(train)
    # thời gian khuyến mãi ảnh hưởng tích luỹ -> thêm số ngày promo liên tục gần đây
    train = add_promo_month(train)
    # đối thủ cạnh tranh làm giảm sales -> thêm số ngày kể từ khi đối thủ mở cửa
    train = add_competition_days(train)
    # Sales lệch phải mạnh -> log1p(Sales) cho phân phối đều hơn, model học tốt hơn
    train = add_log_sales(train)

    if args.verbose:
        print("\n--- Train quality report ---")
        validate(train)

    save(train, f"{args.out_dir}/train_cleaned", args.format)
    save(store, f"{args.out_dir}/store_cleaned", args.format)

    # --- Test pipeline (no outlier removal, no log sales) ---
    if not args.skip_test:
        print("Cleaning test...")
        test = merge_store(test, store)
        test = fill_missing(test)
        test = fix_types(test)
        test = add_holiday_proximity(test)
        test = add_promo_month(test)
        test = add_competition_days(test)

        if args.verbose:
            print("\n--- Test quality report ---")
            validate(test)

        save(test, f"{args.out_dir}/test_cleaned", args.format)

    print("Done.")


if __name__ == "__main__":
    main()
