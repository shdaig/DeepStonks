def get_tickers_from_file(filename: str) -> list[str]:
    with open(filename, 'r') as f:
        tickers = [line[:-1] for line in f.readlines() if len(line[:-1]) != 0]
    return tickers


if __name__ == "__main__":
    tickers = get_tickers_from_file("../../local_data/tickers_test.txt")
    print(f"{tickers}")
