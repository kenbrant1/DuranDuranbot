~ai_assist_instructions_page_func = {

    ~firstview.close;

    ~firstview = View(w, Rect(0, 0, w.bounds.width, w.bounds.height));
    /*view.clearOnRefresh = false;*/
    ~firstview.background = ~backgroundcolor;
    ~firstviewboundsWIDTH = ~firstview.bounds.width;
    ~firstviewboundsHEIGHT = ~firstview.bounds.height;

    ~firstviewtext1 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/2.2857142857143, ~firstviewboundsHEIGHT/43.7, ~firstviewboundsWIDTH/8, ~firstviewboundsHEIGHT/21));
    ~firstviewtext1.background=~backgroundcolor;
    ~firstviewtext1.stringColor=~stringcolor;
    ~firstviewtext1.string = "AI Assist Instructions";
    ~firstviewtext1.align = \topLeft;
    ~firstviewtext1.font = Font(size: ~firstviewboundsWIDTH/48);

    ~firstviewtext2 = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/43.7, ~firstviewboundsWIDTH/16, ~firstviewboundsHEIGHT/21)).states_([["back",~stringcolor,~backgroundcolor],["back",~stringcolor,~backgroundcolor]]).action_({arg button;
        ~firstview.close;
        ~assist_compose_page_func.value;
    });
    ~firstviewtext2.background=~stringcolor;
    ~firstviewtext2.string = "Back";
    ~firstviewtext2.font = Font(size: ~firstviewboundsWIDTH/48);

    ~firstviewtext3 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/1.5652173913043, ~firstviewboundsHEIGHT/43.7, ~firstviewboundsWIDTH/2.88, ~firstviewboundsHEIGHT/4.37));
    ~firstviewtext3.background=~backgroundcolor;
    ~firstviewtext3.stringColor=~stringcolor;
    ~firstviewtext3.string = "*Important Note*... You can kill all sounds and processes by hitting "++"cmd/ctrl - period".quote++"on your keyboard.";
    ~firstviewtext3.align = \topLeft;
    ~firstviewtext3.font = Font(size: ~firstviewboundsWIDTH/65.454545454545);

    ~firstviewtext4 = TextView(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/7.528571428571, ~firstviewboundsWIDTH/1.0285714285714, ~firstviewboundsHEIGHT/1.2));
    ~firstviewtext4.background=~backgroundcolor;
	~firstviewtext4.string = "(Scroll down for more)\n\n\n\n
What works currently -
\n\n New Part, Generate New, and all duplicate functions";
    ~firstviewtext4.stringColor=~stringcolor;
    ~firstviewtext4.font = Font(size: ~firstviewboundsWIDTH/48);

};