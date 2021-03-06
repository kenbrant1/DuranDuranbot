~ai_assist_instructions_page_func = {

    ~firstview.close;

    ~firstview = View(w, Rect(0, 0, w.bounds.width, w.bounds.height));
    /*view.clearOnRefresh = false;*/
    ~firstview.background = ~backgroundcolor;
    ~firstviewboundsWIDTH = ~firstview.bounds.width;
    ~firstviewboundsHEIGHT = ~firstview.bounds.height;

    ~firstviewtext1 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/2.2857142857143, ~firstviewboundsHEIGHT/43.7, ~firstviewboundsWIDTH/6.25, ~firstviewboundsHEIGHT/21));
    ~firstviewtext1.background=~backgroundcolor;
    ~firstviewtext1.stringColor=~stringcolor;
    ~firstviewtext1.string = "Instructions/Help";
    ~firstviewtext1.align = \topLeft;
    ~firstviewtext1.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

    ~firstviewtext2 = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/43.7, ~firstviewboundsWIDTH/16, ~firstviewboundsHEIGHT/21)).states_([["back",~stringcolor,~backgroundcolor],["back",~stringcolor,~backgroundcolor]]).action_({arg button;
        ~firstview.close;
        ~assist_compose_page_func.value;
    });
    ~firstviewtext2.background=~stringcolor;
    ~firstviewtext2.string = "Back";
    ~firstviewtext2.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

    ~firstviewtext3 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/1.5652173913043, ~firstviewboundsHEIGHT/43.7, ~firstviewboundsWIDTH/2.88, ~firstviewboundsHEIGHT/4.37));
    ~firstviewtext3.background=~backgroundcolor;
    ~firstviewtext3.stringColor=~stringcolor;
    ~firstviewtext3.string = "*Important Note*... You can kill all sounds and processes by hitting "++"cmd/ctrl - period".quote++"on your keyboard.";
    ~firstviewtext3.align = \topLeft;
    ~firstviewtext3.font = Font(~default_font, size: ~firstviewboundsWIDTH/65.454545454545);

    ~firstviewtext4 = TextView(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/7.528571428571, ~firstviewboundsWIDTH/1.0285714285714, ~firstviewboundsHEIGHT/1.2));
    ~firstviewtext4.background=~backgroundcolor;
	~firstviewtext4.string = "(Scroll down for more)

DDbot Assist Composition allows you to compose music with the help of DuranDuranbot.

Quick youtube tutorials/demonstrations -
https://www.youtube.com/watch?v=WEjo4IdC0hs
https://www.youtube.com/watch?v=ccNjaQmctoQ

***DDbot Trains Itself As You Compose in Assisted Composition. Any action you take during composition which implies you like something, DDbot will machine learn it. :) As explained in beginning of this YouTube video - https://www.youtube.com/watch?v=tYJANh9uxZM

HOW TO USE -

New Part - creates new part, which has no audio

Set Delete Parts - choose which parts to get rid of audio. does not delete the part though. After choosing parts, delete by hitting the Delete Parts button.

Delete Parts - delete parts after setting delete parts.

Undo Delete Parts - undo deleted parts. must hit Set Delete Parts first in order to use this.

Generate New - Click on part, then click this button to generate new part audio.

Undo Generate New - Click on part, then click this to undo generate new audio.

Avg. part length (secs.) - Set the average length of a part in seconds.

Range (+/-) - Set the +/- range for average part length.

Set Duplicate - choose which part to duplicate. after clicking this button, choose part you wish to duplicate... then choose part to overwrite, then hit the Duplicate button to duplicate.

Duplicate - duplicate part after setting duplicate parts.

Undo Duplicate - undo duplicate parts. must hit Set Duplicate Parts first in order to use this.

Batch Duplicate... To dup. 1st/last - range of parts to duplicate. Dup. into 1st/last - range of parts to be duplicated into. Click Set Dup. to perform the action... and click Undo to undo it.

Merge - Merge two parts together. Beginning part is the beginning part of the merge, and the end part is the ending part of the merge. Merge into part is the part in which the two prior merged parts will go into.

Trim - Trim a part by using the sliders to determine from which bit to which bit to trim... to keep. Everything outside of the range of bits as determined by the sliders will be discarded. Play Part plays the full part as in is currently... Test Trim tests the trimmed part as determined by the sliders, but does not trim the part yet.... and Trim Trims the part. You can undo a Trim by clicking Undo or Set Original.

Machine learning is ON/OFF - If machine learning during the composition process is on or off. Doing things like duplicating parts/etc. or using the Machine Learn Full Piece button will teach DDbot how to compose if on.

Using ML data is ON/OFF - If using machine learning data to compose music on this page is on or off. For instance... when you generate a music for a part. Using no ML data will result in the use of untrained data for generating music for parts.

Load Piece To Main Page - load the entire peice the to the main page.

Machine Learn Full Piece - Train DDbot using the full piece.
";
    ~firstviewtext4.stringColor=~stringcolor;
    ~firstviewtext4.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

};