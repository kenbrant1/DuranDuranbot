~instructions_page_func = {

    ~firstview.close;

    ~firstview = View(w, Rect(0, 0, w.bounds.width, w.bounds.height));
    /*view.clearOnRefresh = false;*/
    ~firstview.background = ~backgroundcolor;
    ~firstviewboundsWIDTH = ~firstview.bounds.width;
    ~firstviewboundsHEIGHT = ~firstview.bounds.height;

    ~firstviewtext1 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/2.2857142857143, ~firstviewboundsHEIGHT/43.7, ~firstviewboundsWIDTH/8, ~firstviewboundsHEIGHT/21));
    ~firstviewtext1.background=~backgroundcolor;
    ~firstviewtext1.stringColor=~stringcolor;
    ~firstviewtext1.string = "Instructions";
    ~firstviewtext1.align = \topLeft;
    ~firstviewtext1.font = Font(size: ~firstviewboundsWIDTH/48);

    ~firstviewtext2 = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/43.7, ~firstviewboundsWIDTH/16, ~firstviewboundsHEIGHT/21)).states_([["back",~stringcolor,~backgroundcolor],["back",~stringcolor,~backgroundcolor]]).action_({arg button;
        ~firstview.close;
        ~main_page_func.value;
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
	~firstviewtext4.string = "(Scroll down for more)\n\nHOW TO PLAY A PIECE\n\nPlay a DuranDuranbot piece by clicking the "++"generate piece".quote++"  button on the Main Page, and then by clicking the "++"play piece".quote++"  button. "++"Random seeds".quote++" help DuranDuranbot generate a musical piece - To generate a new piece, click on the "++"New Seeds".quote++" button to generate new random seeds - or manually set the seeds by clicking on the "++"Set Seeds".quote++" button - and then click on "++"generate piece".quote++" button in both cases. Or... you can quickly generate a new piece based on entirely new random seeds by simply clicking on the "++"quick new piece".quote++" button just below the "++"generate piece".quote++" button. \n\nMACHINE LEARNING\n\nMachine learning is enabled by default, but you can disable it on the main page. Using the machine learning function allows for you to teach DuranDuranbot what you prefer to hear in a piece by rating a piece, and for it become better at composing music for you. On the Main Page, you can rate a piece by clicking on any of the point ratings and clicking "++"Rate".quote++". The rating system is currently set to rate the last "++(~rate_time_range)++" seconds from your current position in the piece designated by where the time slider is positioned... Doing all of this will train DuranDuranbot how to compose music. All training data will be saved to the DuranDuranbot "++"ddbot_training_data".quote++" folder for future use. (Side Note: You can hybridize the training data of numerous DD'bots as well by going into the DD'bot training folder.)\n\nDDbot's machine learning system utilizes mathematical weights to compose/learn to compose music. For instance, a piece compostition variable might be a note interval, and each note interval (1 step, 2 steps) would have a weight assigned to it in a given situation... and DDbot would choose an interval by comparing the weight values of each interval in a given situation, *tending* to choose a higher weighted variable using weighted probability. When you rate a section of a piece while using DDbot, a variable in a given situation is rewarded with points as to increase the value of it's weight. The more points you give a variable, the more chance it will be used in a piece composition. How powerful each xpt rating is - x16 pts is 80% weighted chance for variable in 5 reward instances for that variable... x8 pts is 80% weighted chance for variable in 10 reward instances. x4 pts is 80% weighted chance for variable in 20 reward instances.... and so on in that same pattern. Put simply, a higher rating point value equals faster training of DDbot, but less fine-tuning.\n\nSettings\n\nBy clicking the "++"Settings".quote++" button on the Main Page, you can choose which player(s) you can traget to compose music and be trained.\n\nLOOPING\n\n1st text box is starting bit position, 2nd text box is ending bit position. Last text box is how many loops.\n\nCURRENT BUGS\n\n- \n\n\n\nIf you have any questions, please feel free to contact -\n\nken_brant@ymail.com\n\nor\n\nkenbrant7@gmail.com";
    ~firstviewtext4.stringColor=~stringcolor;
    ~firstviewtext4.font = Font(size: ~firstviewboundsWIDTH/48);

};