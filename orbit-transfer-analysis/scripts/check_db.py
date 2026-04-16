import sys
sys.path.append("/Users/heewon/Desktop/무제 폴더")

from orbit_transfer.database.storage import TrajectoryDatabase

db = TrajectoryDatabase()
rows = db.get_results()

for row in rows:
    print(row)

db.close()