.PHONY: up down logs api mlflow psql

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f --tail=200

api:
	curl -s http://localhost:8000/health | jq . || true

mlflow:
	python -c "import webbrowser; webbrowser.open('http://localhost:5000')"

psql:
	PGPASSWORD=${POSTGRES_PASSWORD} psql -h localhost -p 5433 -U ${POSTGRES_USER} -d ${POSTGRES_DB}
