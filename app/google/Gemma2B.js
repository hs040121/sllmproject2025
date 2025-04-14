// Gemma2B.js

function generateText() {
    const prompt = document.getElementById("userPrompt").value;

    if (!prompt) {
        alert("입력값을 먼저 입력하세요!");
        return;
    }

    $.ajax({
        url: "http://localhost:8000/generate_text",
        type: "POST",
        contentType: "application/json",
        data: JSON.stringify({ prompt: prompt }),

        success: function(response) {
            if (response.result) {
                document.getElementById("result").innerHTML = response.result;
            } else {
                document.getElementById("result").innerHTML = "결과가 없습니다.";
            }
        },
        error: function(error) {
            document.getElementById("result").innerHTML = "에러 발생!";
            console.log(error);
        }
    });
}
