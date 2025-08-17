# nSpaces Backend – API Docs

Base URL: `http://localhost:8000`

Auth: **Bearer JWT** (generated via `/auth/token`)

---

## Authentication

### `POST /auth/token`

Obtain a JWT access token.

**Body (form-data):**

```json
{
  "username": "user@example.com",
  "password": "mypassword"
}
```

**Response:**

```json
{
  "access_token": "jwt-token-here",
  "token_type": "bearer"
}
```

---

## Spaces

### `POST /spaces`

Create a new space.

**Body:**

```json
{
  "name": "My Team Space",
  "description": "Knowledge sharing hub"
}
```

**Response:**

```json
{
  "id": "space_id",
  "name": "My Team Space",
  "description": "Knowledge sharing hub",
  "owner_id": "user123"
}
```

---

### `GET /spaces/{space_id}`

Get details of a space.

**Response:**

```json
{
  "id": "space_id",
  "name": "My Team Space",
  "members": ["user123", "user456"],
  "created_at": "2025-08-14T12:00:00Z"
}
```

---

### `POST /spaces/{space_id}/invite`

Create an invite token.

**Body:**

```json
{
  "expires_in": 3600
}
```

**Response:**

```json
{
  "invite_url": "https://yourapp.com/invite/abc123"
}
```

---

## Messages

### `POST /spaces/{space_id}/messages`

Send a message to a space.

**Body:**

```json
{
  "text": "Hello team!"
}
```

**Response:**

```json
{
  "id": "msg_id",
  "user_id": "user123",
  "text": "Hello team!",
  "created_at": "2025-08-14T12:10:00Z"
}
```

---

### `GET /spaces/{space_id}/messages`

Fetch messages for a space.

**Query params:**

* `limit`: number of messages (default 50)
* `before`: ISO datetime (optional)

**Response:**

```json
[
  {
    "id": "msg_id",
    "user_id": "user123",
    "text": "Hello team!",
    "created_at": "2025-08-14T12:10:00Z"
  }
]
```

---

## Documents & RAG

### `POST /spaces/{space_id}/documents`

Upload a document for RAG.

**Form-data:**

* `files`: uploaded file

**Response:**

```json
{
  "doc_id": "doc123",
  "chunks_indexed": 42
}
```

---

### `POST /spaces/{space_id}/query`

Ask a question over a space’s documents.

**Body:**

```json
{
  "query": "What is our project deadline?"
}
```

**Response:**

```json
{
  "answer": "The project deadline is Sept 15th.",
  "sources": [
    {
      "doc_id": "doc123",
      "chunk_text": "The project deadline is Sept 15th.",
      "score": 0.92
    }
  ]
}
```

---

That’s a **high-level API doc**. FastAPI already exposes interactive docs at:

* Swagger: `/docs`
* ReDoc: `/redoc`

---

Want me to extend this doc into a **machine-readable OpenAPI schema** (`openapi.yaml`) so you can share it with teammates or import into Postman?
