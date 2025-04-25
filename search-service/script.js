async function generateEmbedding() {
    const lastSent = document.getElementById('lastSent');
    const productData = document.getElementById('productData').value;
    const embeddingResult = document.getElementById('embeddingResult');

    embeddingResult.textContent = 'Loading...';

    if (!productData) {
        embeddingResult.textContent = 'Product data is required!';
        return;
    }

    try {
        const response = await fetch('http://localhost:8001/embed', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                texts: [productData]
            })
        });

        if (!response.ok) {
            throw new Error('Failed to fetch embeddings');
        }

        const data = await response.json();
        lastSent.textContent = `Last sent: ${productData}`;
        embeddingResult.textContent = `Embedding:\n${JSON.stringify(data.embeddings, null, 2)}\n\nID: ${data.ids[0]}`;

    } catch (error) {
        embeddingResult.textContent = `Error: ${error.message}`;
    }
}

async function checkCount() {
    const embeddingResult = document.getElementById('embeddingResult');
    embeddingResult.textContent = 'Loading...';

    try {
        const response = await fetch('http://localhost:8001/count');
        if (!response.ok) {
            throw new Error('Failed to fetch count');
        }
        const data = await response.json();
        embeddingResult.textContent = `Document count: ${data.count}`;
    } catch (error) {
        embeddingResult.textContent = `Error: ${error.message}`;
    }
}

async function getAll() {
    const embeddingResult = document.getElementById('embeddingResult');
    embeddingResult.textContent = 'Loading...';

    try {
        const response = await fetch('http://localhost:8001/get_all');
        if (!response.ok) {
            throw new Error('Failed to fetch data');
        }
        const data = await response.json();
        embeddingResult.textContent = `All data:\n${JSON.stringify(data, null, 2)}`;
    } catch (error) {
        embeddingResult.textContent = `Error: ${error.message}`;
    }
}

async function searchDocuments() {
    const searchQuery = document.getElementById('searchQuery').value;
    const searchResults = document.getElementById('searchResults');

    searchResults.textContent = 'Loading...';

    if (!searchQuery) {
        searchResults.textContent = 'Search query is required!';
        return;
    }

    try {
        const response = await fetch(`http://localhost:8001/search?query=${encodeURIComponent(searchQuery)}&top_k=5`);
        
        if (!response.ok) {
            throw new Error('Failed to fetch search results');
        }

        const data = await response.json();
        const resultsText = data.results.map((doc, idx) => `Result ${idx + 1}: ${doc}`).join("\n");
        searchResults.textContent = `Search Results:\n${resultsText} \n data: ${JSON.stringify(data, null, 2)}`;

    } catch (error) {
        searchResults.textContent = `Error: ${error.message}`;
    }
}