// You can add functionality to buttons here, for example:
document.querySelectorAll('.btn').forEach(button => {
    button.addEventListener('click', function () {
        const buttonText = this.innerText;
        alert(`You clicked the "${buttonText}" button.`);
    });
});
