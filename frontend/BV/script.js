document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('divineForm');
    const outputSection = document.getElementById('outputSection');
    const outputText = document.getElementById('outputText');
    const audioBtn = document.getElementById('playAudioBtn');
    
    // Read URL parameters and populate the input field
    const urlParams = new URLSearchParams(window.location.search);
    const textParam = urlParams.get('text');
    if (textParam) {
        document.getElementById('userName').value = decodeURIComponent(textParam);
    }
    
    let currentMessage = "";
    
    form.addEventListener('submit', async(e) => {
        e.preventDefault();
        
        const name = document.getElementById('userName').value;
        const lang = document.getElementById('userLanguage').value || 'en-US';
        
        // Generate a divine message
        let data = null;
        try {
                // const response = await fetch("http://127.0.0.1:8000/get-meaning", {
                const response = await fetch("https://gitaversebackend.onrender.com/get-meaning", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    shloka: name,
                    language: lang
                })
            });

            data = await response.json();
            if (data.text) {
                currentMessage = data.text; 
            }
        } catch (err) {
            console.error("FastAPI error:", err);
        }
        // Reveal the output section
        outputSection.style.visibility = 'visible';
        
        // Start Typewriter Animation
        typeWriter(currentMessage, outputText);
        
        // Setup Audio Button
        // Remove old event listeners to prevent duplicates
        const newAudioBtn = audioBtn.cloneNode(true);
        audioBtn.parentNode.replaceChild(newAudioBtn, audioBtn);
        
        newAudioBtn.style.display = "none";

        if (data && data.audio_url) {
            // 🔴 ADD THIS
            const audio = document.createElement("audio");
             // audio.src = "http://127.0.0.1:8000" + data.audio_url;
             audio.src = "https://gitaversebackend.onrender.com" + data.audio_url;
            audio.preload = "auto";

    // 🔴 IMPORTANT: attach to DOM (invisible)
           document.body.appendChild(audio);

           newAudioBtn.addEventListener('click', () => {
                console.log("Listen button clicked");
                audio.currentTime = 0;
                audio.play().catch(err => console.error(err));
          });
        }
    });

    function typeWriter(text, element) {
        element.textContent = '';
        let i = 0;
        const speed = 50; // Speed in ms
        
        function type() {
            if (i < text.length) {
                element.textContent += text.charAt(i);
                i++;
                setTimeout(type, speed);
            }
        }
        type();
    }

    function speakText(text, lang) {
        if ('speechSynthesis' in window) {
            // Cancel any ongoing speech
            window.speechSynthesis.cancel();

            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = lang;
            utterance.rate = 0.85; // Slightly slower for a divine feel
            utterance.pitch = 1.0;
            
            window.speechSynthesis.speak(utterance);
        } else {
            alert("Divine speech is not supported in this browser.");
        }
    }

    // --- Particle System for Dust Motes ---
    const canvas = document.getElementById('dustCanvas');
    const ctx = canvas.getContext('2d');
    let particles = [];
    
    let mouse = {
        x: null,
        y: null,
        radius: 100
    };

    window.addEventListener('mousemove', (e) => {
        mouse.x = e.clientX;
        mouse.y = e.clientY;
    });

    function resizeCanvas() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        initParticles();
    }

    class Particle {
        constructor() {
            this.x = Math.random() * canvas.width;
            this.y = Math.random() * canvas.height;
            
            // Depth Simulation: 3 Layers
            this.depth = Math.random(); // 0 to 1
            
            if (this.depth > 0.6) {
                // Foreground: Larger, faster, brighter
                this.size = Math.random() * 2 + 1.5;
                this.speedMultiplier = 0.7;
                this.baseOpacity = 0.7;
            } else if (this.depth > 0.3) {
                // Midground
                this.size = Math.random() * 1.5 + 0.5;
                this.speedMultiplier = 0.4;
                this.baseOpacity = 0.5;
            } else {
                // Background: Tiny, slow, dim
                this.size = Math.random() * 1 + 0.2;
                this.speedMultiplier = 0.2;
                this.baseOpacity = 0.3;
            }

            this.speedX = (Math.random() - 0.5) * this.speedMultiplier;
            this.speedY = (Math.random() - 0.5) * this.speedMultiplier;
            
            // Twinkle Properties
            this.opacity = this.baseOpacity;
            this.twinkleSpeed = Math.random() * 0.01 + 0.002;
            this.twinkleDir = Math.random() > 0.5 ? 1 : -1;
        }
        
        update() {
            // Twinkle Logic
            this.opacity += this.twinkleSpeed * this.twinkleDir;
            if (this.opacity >= this.baseOpacity + 0.2 || this.opacity >= 1) {
                this.opacity = Math.min(this.opacity, 1);
                this.twinkleDir = -1;
            } else if (this.opacity <= this.baseOpacity - 0.2 || this.opacity <= 0.1) {
                this.opacity = Math.max(this.opacity, 0.1);
                this.twinkleDir = 1;
            }

            // Mouse Repulsion
            if (mouse.x != null) {
                let dx = this.x - mouse.x;
                let dy = this.y - mouse.y;
                let distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance < mouse.radius && distance > 0) {
                    const forceDirectionX = dx / distance;
                    const forceDirectionY = dy / distance;
                    const force = (mouse.radius - distance) / mouse.radius;
                    this.x += forceDirectionX * force * 2;
                    this.y += forceDirectionY * force * 2;
                }
            }

            this.x += this.speedX;
            this.y += this.speedY;

            // Wrap around screen edges
            if (this.x < 0) this.x = canvas.width;
            if (this.x > canvas.width) this.x = 0;
            if (this.y < 0) this.y = canvas.height;
            if (this.y > canvas.height) this.y = 0;
        }
        
        draw() {
            ctx.fillStyle = `rgba(212, 175, 55, ${this.opacity})`; // Gold color
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    function initParticles() {
        particles = [];
        const particleCount = Math.min(window.innerWidth * 0.5, 400); // Increased density
        for (let i = 0; i < particleCount; i++) {
            particles.push(new Particle());
        }
    }

    function animateParticles() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        particles.forEach(p => {
            p.update();
            p.draw();
        });
        requestAnimationFrame(animateParticles);
    }

    window.addEventListener('resize', resizeCanvas);
    resizeCanvas(); // Initial setup
    animateParticles();
});