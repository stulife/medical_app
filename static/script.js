class MedicalQAApp {
    constructor() {
        this.sessionId = this.generateSessionId();
        this.lastQuestion = null;
        this.lastAnswer = null;
        this.initElements();
        this.attachEventListeners();
    }

    initElements() {
        this.chatHistory = document.getElementById('chat-history');
        this.questionInput = document.getElementById('question-input');
        this.sendBtn = document.getElementById('send-btn');
        this.feedbackButtons = document.querySelectorAll('.feedback-btn');
    }

    attachEventListeners() {
        this.sendBtn.addEventListener('click', () => this.askQuestion());
        this.questionInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.askQuestion();
            }
        });

        this.feedbackButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const score = parseInt(e.target.dataset.score);
                this.submitFeedback(score);
            });
        });
    }

    generateSessionId() {
        return 'session_' + Math.random().toString(36).substr(2, 9);
    }

    async askQuestion() {
        const question = this.questionInput.value.trim();
        if (!question) return;

        // 显示用户问题
        this.displayMessage(question, 'user');
        this.questionInput.value = '';

        try {
            // 发送请求到后端
            const response = await fetch('/api/qa/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: question,
                    session_id: this.sessionId
                })
            });

            const data = await response.json();
            
            if (data.error) {
                this.displayMessage(`错误: ${data.error}`, 'bot');
            } else {
                this.lastQuestion = data.question;
                this.lastAnswer = data.answer;
                this.displayMessage(`${data.answer} (评分: ${data.score})`, 'bot');
            }
        } catch (error) {
            this.displayMessage(`请求失败: ${error.message}`, 'bot');
        }
    }

    async submitFeedback(score) {
        if (!this.lastQuestion || !this.lastAnswer) {
            alert('没有可评分的回答');
            return;
        }

        try {
            const response = await fetch('/api/qa/feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: this.lastQuestion,
                    answer: this.lastAnswer,
                    score: score,
                    session_id: this.sessionId
                })
            });

            const data = await response.json();
            
            if (data.error) {
                alert(`反馈提交失败: ${data.error}`);
            } else {
                alert('感谢您的反馈！');
            }
        } catch (error) {
            alert(`反馈提交失败: ${error.message}`);
        }
    }

    displayMessage(message, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${sender}-message`);
        messageDiv.textContent = message;
        this.chatHistory.appendChild(messageDiv);
        this.chatHistory.scrollTop = this.chatHistory.scrollHeight;
    }
}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    new MedicalQAApp();
});