/**
 * ==========================================================================
 * SOCIAL MEDIA USAGE RESEARCH PROJECT - MAIN JAVASCRIPT
 * Description: Vanilla JavaScript for UI interactions, navigation, 
 *              and responsive controls. Zero dependencies.
 * ==========================================================================
 */

// Wait until the HTML document is fully loaded before executing scripts
document.addEventListener("DOMContentLoaded", () => {

    /* ----------------------------------------------------------------------
       1. Mobile Navigation Menu Toggle
       ---------------------------------------------------------------------- */
    const menuToggleBtn = document.querySelector(".menu-toggle");
    const navLinks = document.querySelector(".nav-links");

    if (menuToggleBtn && navLinks) {
        menuToggleBtn.addEventListener("click", () => {
            navLinks.classList.toggle("show");
            
            // Toggle aria-expanded for screen-reader accessibility
            const isExpanded = navLinks.classList.contains("show");
            menuToggleBtn.setAttribute("aria-expanded", isExpanded);
        });

        // Close mobile menu when a navigation link is clicked
        navLinks.querySelectorAll("a").forEach(link => {
            link.addEventListener("click", () => {
                if (navLinks.classList.contains("show")) {
                    navLinks.classList.remove("show");
                    menuToggleBtn.setAttribute("aria-expanded", "false");
                }
            });
        });
    }

    /* ----------------------------------------------------------------------
       2. Active Navigation Highlighting on Scroll
       Uses the modern browser IntersectionObserver API to detect which
       section is currently in the viewport and update navbar links.
       ---------------------------------------------------------------------- */
    const sections = document.querySelectorAll("section[id]");
    const navItems = document.querySelectorAll(".nav-links a");

    if ("IntersectionObserver" in window && sections.length > 0) {
        const observerOptions = {
            root: null,
            rootMargin: "-20% 0px -60% 0px", // Trigger when section is in upper-mid viewport
            threshold: 0
        };

        const sectionObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const currentId = entry.target.getAttribute("id");
                    
                    navItems.forEach(link => {
                        if (link.getAttribute("href") === `#${currentId}`) {
                            link.classList.add("active");
                        } else {
                            link.classList.remove("active");
                        }
                    });
                }
            });
        }, observerOptions);

        sections.forEach(section => sectionObserver.observe(section));
    }

    /* ----------------------------------------------------------------------
       3. Interactive Topic Pillar Cards
       Clicking a topic card toggles an active state and smoothly scrolls
       the user to the corresponding research question and variables.
       ---------------------------------------------------------------------- */
    const topicCards = document.querySelectorAll(".topic-card");

    topicCards.forEach(card => {
        card.addEventListener("click", () => {
            // Remove active state from all cards, then activate clicked card
            topicCards.forEach(c => c.classList.remove("active"));
            card.classList.add("active");

            // Optional educational visual feedback: briefly pulse related variable tags
            const topicType = card.getAttribute("data-topic");
            highlightVariablesForTopic(topicType);
        });
    });

    /**
     * Highlights corresponding variable badges when a topic card is selected.
     * @param {string} topic - The data-topic attribute key
     */
    function highlightVariablesForTopic(topic) {
        const variableTags = document.querySelectorAll(".variable-tag");
        
        variableTags.forEach(tag => {
            const tagTopics = tag.getAttribute("data-topics") || "";
            if (topic && tagTopics.includes(topic)) {
                tag.style.borderColor = "var(--primary-color)";
                tag.style.backgroundColor = "var(--primary-light)";
            } else {
                tag.style.borderColor = "var(--border-color)";
                tag.style.backgroundColor = "var(--bg-alt)";
            }
        });
    }

    /* ----------------------------------------------------------------------
       4. Log confirmation for development
       ---------------------------------------------------------------------- */
    console.log("Research website loaded successfully. Running in local mode.");
});
