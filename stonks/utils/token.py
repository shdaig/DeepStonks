def get_token(filename: str) -> str:
    with open(filename, 'r') as f:
        token_line = f.readlines()[0][:-1]

    return token_line


if __name__ == "__main__":
    token = get_token("../../local_data/token.txt")
    print(f"Token: {token}")
