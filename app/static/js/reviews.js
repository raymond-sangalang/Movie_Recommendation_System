// reviews.js - handles front-end logistics when loading movie reviews
//            as well as submitting movie reviews

(function () {
    
    let movieId = null;
    let userId = null;
    let reviews = [];         

    //  filters type of sort: latest or rating 
    let currentSort = "newest";         

    document.addEventListener("DOMContentLoaded", () => {

        const root = document.getElementById("review-root");
        
        if (!root)  return;        // No review section on the page
            
        // Set variables given by the users review form
        movieId = root.dataset.movieId;
        userId = root.dataset.userId || null;

     
        const form = document.getElementById("review-form");
        if (form) 
            form.addEventListener("submit", handleSubmitReview);
        
        // Filtering the Sort button
        const sortRatingBtn = document.getElementById("sort-rating");
        const sortNewestBtn = document.getElementById("sort-newest");

        if (sortRatingBtn) {
            sortRatingBtn.addEventListener("click", () => {
                currentSort = "rating";
                renderReviews();
            });
        }

        if (sortNewestBtn) {
            sortNewestBtn.addEventListener("click", () => {
                currentSort = "newest";
                renderReviews();
            });
        }

        if (movieId)                // Initial load
            loadReviews();
    });


    async function loadReviews() {

        try 
        {
            const res = await fetch(`/api/reviews/${movieId}`);
            if (!res.ok) {
                console.error("Failed to load reviews:", res.status);
                return;
            }

            const data = await res.json();
            reviews = data.reviews || [];

            renderAverageRating(data.average_rating);
            renderReviews();
        } 
        catch (err) {
            console.error("Error fetching reviews:", err);
        }
    }


    function renderAverageRating(avg) {

        const avgContainer = document.getElementById("avg-rating");
       
        if (!avgContainer) return;

        if (avg == null) {
            avgContainer.innerHTML = "<em>No ratings yet. Be the first to review!</em>";
            return;
        }

        const starsHtml = buildStars(avg);
        avgContainer.innerHTML = `
            <div>
                <strong>Average Rating:</strong>
                <span class="stars">${starsHtml}</span>
                <span style="font-size:14px; margin-left:6px;">(${avg.toFixed(2)}/5.0)</span>
            </div>
        `;
    }


    function renderReviews() {
        const list = document.getElementById("reviews-list");
        if (!list) return;

        if (!reviews.length) {
            list.innerHTML = "<p><em>No reviews yet.</em></p>";
            return;
        }

        let sorted = [...reviews];

        if (currentSort === "rating") {
            sorted.sort((a, b) => b.rating - a.rating);
        } else if (currentSort === "newest") {
            sorted.sort((a, b) => {
                // sort by timestamp desc if available
                const ta = Date.parse(a.timestamp || "") || 0;
                const tb = Date.parse(b.timestamp || "") || 0;
                return tb - ta;
            });
        }

        const itemsHtml = sorted
            .map((r) => {
                const starsHtml = buildStars(r.rating);
                const ts = r.timestamp ? new Date(r.timestamp).toLocaleString() : "";
                return `
                    <div class="container" style="margin-bottom:10px; padding:10px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <strong>User ${r.userId}</strong>
                            </div>
                            <div>
                                <span class="stars">${starsHtml}</span>
                                <span style="font-size:13px; margin-left:6px;">(${Number(r.rating).toFixed(1)})</span>
                            </div>
                        </div>
                        ${r.review ? `<p style="margin-top:8px;">${escapeHtml(r.review)}</p>` : ""}
                        ${ts ? `<div style="font-size:12px; color:#666; margin-top:4px;">${ts}</div>` : ""}
                    </div>
                `;
            })
            .join("");

        list.innerHTML = itemsHtml;
    }


    function buildStars(rating) {
        // Using javascripts logistics/operators to calculate rating stats
       
        const value = Number(rating) || 0; 
        const full = Math.floor(value);
        const half = value - full >= 0.5 ? 1 : 0;
        const empty = 5 - full - half;

        let out = "";
        for (let i = 0; i < full; i++) 
            out += "★";

        if (half) 
            out += "☆";

        for (let i = 0; i < empty; i++) 
            out += "✩";

        return out;
    }


    async function handleSubmitReview(event) {

        event.preventDefault();

        const ratingElement = document.getElementById("review-rating");
        const textElement = document.getElementById("review-text");
        const userSelect = document.getElementById("review-user-id");

        if (!ratingElement || !textElement) return;

        const rating = ratingElement.value;
        const review = textElement.value.trim();

        if (!rating) {
            alert("Please select a rating.");
            return;
        }

        // Condition: to check if user already reviewed
        const uid = userSelect ? userSelect.value : (userId || 0);      
               

        try {
            const res = await fetch(`/api/reviews/${movieId}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    user_id:    uid, 
                                rating, 
                                review,
                }),
            });

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                alert(errData.error || "Failed to submit review.");
                return;
            }

            // Clear the form and reload reviews
            ratingElement.value = "5";
            textElement.value = "";
            await loadReviews();  
        } 
        catch (err) {
            console.error("Error submitting review:", err);
            alert("Error submitting review.");
        }
    }


    function escapeHtml(str) {

        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

})();
