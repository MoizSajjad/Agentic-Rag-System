from backend.chroma_setup import init_chroma_client, ensure_corrections_collection
from backend import config


def main() -> None:
    client = init_chroma_client()
    try:
        client.delete_collection(name=config.CHROMA_COLLECTIONS.corrections_memory)
    except Exception:
        pass
    ensure_corrections_collection(client)
    print("Corrections memory cleared.")


if __name__ == "__main__":
    main()

