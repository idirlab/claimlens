function submitForm() {
    $('#results-container').empty();
    $('#loader').show();

    const userInput = document.getElementById("input-text").value;
    if (userInput === "") {
        alert("Please input a claim to check!");
        return;
    }

    //  Create AJAX call to send user input to server
    $.ajax({
        type: "GET",
        url: "https://idir.uta.edu/claimlens/submit",
        data: { query: userInput },
        success: function (response) {
            document.getElementById("results-container").innerHTML = response;
            $("html,body").animate(
                {
                    scrollTop: $("#input-text").offset().top,
                },
                "slow"
            );

            var popupArgs = {
                inline: true,
                hoverable: true,
                position: "top center",
                delay: {
                    show: 0,
                    hide: 0,
                },
            };
            
            // .fe-agent, .fe-issue, .fe-side, .fe-position, .fe-frequency, .fe-time, .fe-place, .fe-support_rate
            $('.fe-agent').popup({...popupArgs, target: '.fe-agent'});
            $('.fe-issue').popup({...popupArgs, target: '.fe-issue'});
            $('.fe-side').popup({...popupArgs, target: '.fe-side'});
            $('.fe-position').popup({...popupArgs, target: '.fe-position'});
            $('.fe-frequency').popup({...popupArgs, target: '.fe-frequency'});
            $('.fe-time').popup({...popupArgs, target: '.fe-time'});
            $('.fe-place').popup({...popupArgs, target: '.fe-place'});
            $('.fe-support_rate').popup({...popupArgs, target: '.fe-support_rate'});

            $('.alignment').popup({
                inline: true,
                hoverable: true,
                position: 'left center',
                delay: {
                    show: 0,
                    hide: 0
                },
                // Fix overflow
                onShow: function () {
                    $('.ui.popup').css('width', '400px');
                }
            });

            $('#loader').hide();

        },
        error: function (error) {
            console.log(error);
        }
    });
}

function toggleSummary(e) {
    $(e).parent().find('.bill-summary').toggleClass('hide-long-text');
    if ($(e).text() == 'Show more') {
        $(e).text('Show less');
    } else {
        $(e).text('Show more');
    }
}
