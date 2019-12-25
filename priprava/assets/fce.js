var target = $('#graph')[0]
    target.addEventListener('mousemove', function(evt) {
        setProps({ 
            'event': {'x':evt.x, 
                      'y':evt.y }
        })
        console.log(evt)
    })
    console.log(this)
