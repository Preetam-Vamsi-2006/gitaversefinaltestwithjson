// Chapter data
const chapters = [
{ number: 1, name: "Arjun Vishad Yog", sanskrit: "अर्जुनविषादयोग", verses: 47, description: "Lamenting the Consequences of War" },
{ number: 2, name: "Sānkhya Yog", sanskrit: "सांख्ययोग", verses: 72, description: "The Yog of Analytical Knowledge" },
{ number: 3, name: "Karm Yog", sanskrit: "कर्मयोग", verses: 43, description: "The Yog of Action" },
{ number: 4, name: "Gyan Karm Sanyās Yog", sanskrit: "ज्ञानकर्मसंन्यासयोग", verses: 42, description: "The Yog of Knowledge and Renunciation of Action" },
{ number: 5, name: "Karm Sanyās Yog", sanskrit: "कर्मसंन्यासयोग", verses: 29, description: "The Yog of Renunciation" },
{ number: 6, name: "Dhyān Yog", sanskrit: "ध्यानयोग", verses: 47, description: "The Yog of Meditation" },
{ number: 7, name: "Gyan Vigyān Yog", sanskrit: "ज्ञानविज्ञानयोग", verses: 30, description: "The Yog of Knowledge and Wisdom" },
{ number: 8, name: "Akshar Brahm Yog", sanskrit: "अक्षरब्रह्मयोग", verses: 28, description: "The Yog of the Imperishable Absolute" },
{ number: 9, name: "Rāj Vidyā Yog", sanskrit: "राजविद्याराजगुह्ययोग", verses: 34, description: "The Yog of Royal Knowledge and Royal Secret" },
{ number: 10, name: "Vibhūti Yog", sanskrit: "विभूतियोग", verses: 42, description: "The Yog of Divine Glories" },
{ number: 11, name: "Vishwarūp Darshan Yog", sanskrit: "विश्वरूपदर्शनयोग", verses: 55, description: "The Yog of the Vision of the Universal Form" },
{ number: 12, name: "Bhakti Yog", sanskrit: "भक्तियोग", verses: 20, description: "The Yog of Devotion" },
{ number: 13, name: "Kshetra Kshetragya Vibhāg Yog", sanskrit: "क्षेत्रक्षेत्रज्ञविभागयोग", verses: 34, description: "The Yog of Distinction between Field and Knower" },
{ number: 14, name: "Gunatraya Vibhāg Yog", sanskrit: "गुणत्रयविभागयोग", verses: 27, description: "The Yog of the Three Gunas" },
{ number: 15, name: "Purushottam Yog", sanskrit: "पुरुषोत्तमयोग", verses: 20, description: "The Yog of the Supreme Person" },
{ number: 16, name: "Daivāsur Sampad Vibhāg Yog", sanskrit: "दैवासुरसम्पद्विभागयोग", verses: 24, description: "The Yog of the Divine and Demoniac Natures" },
{ number: 17, name: "Shraddhātraya Vibhāg Yog", sanskrit: "श्रद्धात्रयविभागयोग", verses: 28, description: "The Yog of Threefold Faith" },
{ number: 18, name: "Moksha Sanyās Yog", sanskrit: "मोक्षसंन्यासयोग", verses: 78, description: "The Yog of Liberation through Renunciation" }
];

// Generate chapter cards
function generateChapterCards() {
    const container = document.querySelector('.chapters-container');
    
    chapters.forEach(chapter => {
        const card = document.createElement('div');
        card.className = 'chapter-card';
        card.onclick = () => navigateToChapter(chapter.number);
        
        card.innerHTML = `
            <div class="chapter-number">${chapter.number}</div>
            <div class="chapter-header">
                <h2 class="chapter-name">${chapter.name}</h2>
                <p class="chapter-sanskrit">${chapter.sanskrit}</p>
                <p class="chapter-verses">${chapter.verses} verses</p>
            </div>
            <p class="chapter-description">${chapter.description}</p>
        `;
        
        container.appendChild(card);
    });
}

// Navigate to individual chapter page
function navigateToChapter(chapterNumber) {

    const url = `chapter.html?chapter=${chapterNumber}`;

    // open in new tab
    window.open(url, "_blank");

}

// Navbar scroll effect - Optimized with RAF
let scrollRAF;
let lastScrollY = 0;

function updateNavbar() {
    const navbar = document.querySelector('.navbar');
    if (window.scrollY > 50) {
        navbar.classList.add('scrolled');
    } else {
        navbar.classList.remove('scrolled');
    }
    scrollRAF = null;
}

window.addEventListener('scroll', () => {
    if (!scrollRAF) {
        scrollRAF = requestAnimationFrame(updateNavbar);
    }
}, { passive: true });

// Divine cursor - Optimized with RAF
const cursor = document.querySelector('.divine-cursor');

let mouseX = 0, mouseY = 0;
let cursorRAF;

function updateCursor() {
    cursor.style.transform = `translate(${mouseX}px, ${mouseY}px) translate(-50%, -50%)`;
    cursorRAF = null;
}

document.addEventListener('mousemove', (e) => {
    mouseX = e.clientX;
    mouseY = e.clientY;
    if (!cursorRAF) {
        cursorRAF = requestAnimationFrame(updateCursor);
    }
});

document.addEventListener('mousedown', () => {
    cursor.classList.add('click');
});

document.addEventListener('mouseup', () => {
    cursor.classList.remove('click');
});

// Particle effect (dust canvas) - Optimized
const canvas = document.getElementById('dustCanvas');
const ctx = canvas.getContext('2d', { alpha: true });

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

const particles = [];
const particleCount = window.innerWidth < 768 ? 50 : 80; // Reduced from 100

class Particle {
    constructor() {
        this.x = Math.random() * canvas.width;
        this.y = Math.random() * canvas.height;
        this.size = Math.random() * 2 + 0.5;
        this.speedX = Math.random() * 0.5 - 0.25;
        this.speedY = Math.random() * 0.5 - 0.25;
        this.opacity = Math.random() * 0.5 + 0.2;
    }

    update() {
        this.x += this.speedX;
        this.y += this.speedY;

        if (this.x > canvas.width) this.x = 0;
        if (this.x < 0) this.x = canvas.width;
        if (this.y > canvas.height) this.y = 0;
        if (this.y < 0) this.y = canvas.height;
    }

    draw() {
        ctx.fillStyle = `rgba(212, 175, 55, ${this.opacity})`;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fill();
    }
}

function initParticles() {
    for (let i = 0; i < particleCount; i++) {
        particles.push(new Particle());
    }
}

let animationId;
function animateParticles() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    for (let i = 0; i < particles.length; i++) {
        particles[i].update();
        particles[i].draw();
    }

    animationId = requestAnimationFrame(animateParticles);
}

// Handle window resize with debounce
let resizeTimeout;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }, 150);
});

// Pause when tab is hidden
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        cancelAnimationFrame(animationId);
    } else {
        animateParticles();
    }
});

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    generateChapterCards();
    initParticles();
    animateParticles();

    /* ===== MOBILE HAMBURGER MENU ===== */

    const hamburger = document.getElementById("hamburger");
    const navMenu = document.getElementById("nav-menu");

    if (hamburger && navMenu) {
        hamburger.addEventListener("click", () => {
            navMenu.classList.toggle("active");
        });
    }
});
