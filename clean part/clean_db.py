import sqlite3

def clean_database():
    try:
        con = sqlite3.connect("fraud_detection.db")
        cur = con.cursor()
        
        # Delete claims that have no image_paths (old buggy claims)
        cur.execute("DELETE FROM claims WHERE image_paths IS NULL OR image_paths = '[]' OR image_paths = ''")
        deleted = cur.rowcount
        
        con.commit()
        print(f"✅ Successfully deleted {deleted} incomplete/old claims from the database.")
        
    except Exception as e:
        print(f"❌ Error cleaning database: {e}")
    finally:
        con.close()

if __name__ == "__main__":
    clean_database()
