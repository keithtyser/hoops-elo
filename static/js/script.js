/**
 * NCAA Basketball ELO Rating System - UI Functions
 */

// Wait for the document to load
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Ensure no alerts are auto-dismissed by removing any Bootstrap auto-dismissal functionality
    const alerts = document.querySelectorAll('.alert:not(.alert-permanent)');
    
    alerts.forEach(function(alert) {
        // Add the permanent class to all alerts to prevent auto-dismissal
        alert.classList.add('alert-permanent');
        
        // Remove any data attributes that might cause auto-dismissal
        if (alert.hasAttribute('data-bs-dismiss')) {
            alert.removeAttribute('data-bs-dismiss');
        }
        
        // Remove any auto-dismiss functionality from close buttons
        const closeButtons = alert.querySelectorAll('.btn-close');
        closeButtons.forEach(function(button) {
            button.addEventListener('click', function(e) {
                e.preventDefault();
                alert.style.display = 'none';
            });
        });
    });
    
    // Prevent any form from automatically submitting or refreshing
    const forms = document.querySelectorAll('form');
    forms.forEach(function(form) {
        form.addEventListener('submit', function(e) {
            // Allow normal form submission but ensure no auto-refresh
            const submitButton = form.querySelector('button[type="submit"]');
            if (submitButton) {
                submitButton.disabled = true;
                submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
            }
        });
    });
    
    // Add hover class to dashboard cards
    const dashboardCards = document.querySelectorAll('.card.h-100');
    dashboardCards.forEach(function(card) {
        card.classList.add('dashboard-card');
    });
    
    // Form validation
    const formsValidation = document.querySelectorAll('.needs-validation');
    formsValidation.forEach(function(form) {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });
    
    // Score validation for update form
    const scoreInputs = document.querySelectorAll('#score_a, #score_b');
    if (scoreInputs.length > 0) {
        scoreInputs.forEach(function(input) {
            input.addEventListener('change', validateScores);
        });
    }
});

/**
 * Validates that the winning score is higher than the losing score
 */
function validateScores() {
    const scoreA = document.getElementById('score_a');
    const scoreB = document.getElementById('score_b');
    
    if (scoreA && scoreB) {
        const winnerScore = parseInt(scoreA.value) || 0;
        const loserScore = parseInt(scoreB.value) || 0;
        
        if (winnerScore <= loserScore) {
            scoreA.setCustomValidity('Winning score must be higher than losing score');
            scoreB.setCustomValidity('Losing score must be lower than winning score');
        } else {
            scoreA.setCustomValidity('');
            scoreB.setCustomValidity('');
        }
    }
}

/**
 * Formats a number with 1 decimal place
 * @param {number} number - The number to format
 * @returns {string} - The formatted number
 */
function formatRating(number) {
    return number.toFixed(1);
}

/**
 * Applies rating-based styling to elements
 */
function applyRatingStyles() {
    const ratingElements = document.querySelectorAll('[data-rating]');
    ratingElements.forEach(function(element) {
        const rating = parseFloat(element.dataset.rating);
        
        if (rating >= 1650) {
            element.classList.add('bg-rating-high');
        } else if (rating >= 1500) {
            element.classList.add('bg-rating-medium');
        } else {
            element.classList.add('bg-rating-low');
        }
    });
} 