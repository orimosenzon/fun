# Course Tools: setup and user guide

This guide explains how to add a small helper, called **Course tools**, to the course spreadsheet, and how to use it.
You do not need any technical knowledge. Follow the steps in order and do exactly what each step says.
The setup takes about 15 minutes, and you only do it **once**.

---

## What does it do?

Right now, for every new course you copy the task list into Google Tasks and build a shared calendar for the trainer by hand.
Course tools does both for you, straight from the course tab in the spreadsheet.

After setup, you will see a new menu called **Course tools** at the top of the spreadsheet, next to *Help*.
When you open a course tab and pick **Create both** from that menu:

1. **A Google Tasks list** is created with the same name as the tab. It has one task for every row that has a date.
   Each task has its due date, and the notes from the sheet are copied into the task.
   Tasks from yellow rows (the trainer's tasks) start with `[Trainer]`.
2. **A Google Calendar** is created with the same name as the tab. It has one all-day event on each deadline of the trainer (the yellow rows),
   plus a "Course starts" event and a "Course ends" event.
   Every deadline has reminders: an **email 3 days before** and **1 day before**, and a **pop-up 1 day before**, all at 9:00 in the morning.
   The calendar is **shared with the trainer** automatically.

The helper only reads the dates that are already in the sheet. It does not change your tasks or dates.
It writes just two small things into the tab: the trainer's email address and a link to the calendar (see [What the helper writes into the tab](#what-the-helper-writes-into-the-tab)).

---

## Before you start

You need:

- **A computer** (not a phone or tablet). The setup screens do not work well on a phone.
- **Google Chrome** or any other normal web browser.
- To be **signed in to your own Google account**, the one that owns the course spreadsheet.
- About **15 minutes** without interruptions.

> **Tip:** Open this guide in one browser window and the spreadsheet in another, so you can read and work side by side.

A note on keyboard shortcuts. This guide uses **Ctrl** (for example, Ctrl+A).
**On a Mac, use the Cmd key (⌘) instead of Ctrl.**

| Shortcut | What it does |
|---|---|
| Ctrl+A | Select all the text |
| Ctrl+C | Copy |
| Ctrl+V | Paste |
| Ctrl+S | Save |

---

## Part 1: Install the helper (one time only)

The helper is made of two pieces of text: the **script** (the instructions) and the **manifest** (the list of Google services the script uses).
You will copy both from this web page and paste them into your spreadsheet.

### Step 1: Open the script editor

1. Open the **course spreadsheet**.
2. In the menu at the top, click **Extensions**.
3. Click **Apps Script**.

A new browser tab opens. This is the **Apps Script editor**, the place where the helper will live.
It belongs to your spreadsheet only, and nobody else can see it unless they can edit your spreadsheet.

You will see a file called **Code.gs** with a few lines already in it, like this:

```
function myFunction() {

}
```

### Step 2: Give the project a name

1. At the top left, click on the words **Untitled project**.
2. Type: `Course tools`
3. Click **Rename**.

(Google will show this name later, when it asks for permission, so it helps to recognize it.)

### Step 3: Paste the script

1. Open this link in a **new browser tab**:
   **https://raw.githubusercontent.com/orimosenzon/fun/master/vibe_coding/yoram/apps_script/Code.gs**
   You will see a page of plain text that starts with `/**` and `Course tools`.
2. Click anywhere on that text, press **Ctrl+A** to select all of it, then press **Ctrl+C** to copy it.
3. Go back to the **Apps Script editor** tab.
4. Click inside the big white area where `function myFunction()` is written.
5. Press **Ctrl+A** to select everything that is there, then press **Ctrl+V** to paste.
   The old lines are replaced by the long script you copied. That is correct.
6. Press **Ctrl+S** to save. (Or click the small **disk icon** 💾 above the code.)

### Step 4: Show the manifest file

The manifest is hidden by default, so first you need to make it visible.

1. On the **far left edge** of the Apps Script editor there is a narrow column of icons.
   Click the **gear icon ⚙️** (when you move the mouse over it, it says **Project Settings**).
2. Find the checkbox **Show "appsscript.json" manifest file in editor** and **tick it** ✅.
3. Click the **`< >` icon** at the top of that left column (it says **Editor**) to go back to the code.

Now the **Files** list on the left shows two files: `appsscript.json` and `Code.gs`.

### Step 5: Paste the manifest

> ⚠️ **Please don't skip this step.** Without it, the helper stops with an error that says `Tasks is not defined`,
> and Google asks for permission three separate times instead of once.

1. Open this link in a **new browser tab**:
   **https://raw.githubusercontent.com/orimosenzon/fun/master/vibe_coding/yoram/apps_script/appsscript.json**
   You will see a short text that starts with `{` and contains `"timeZone": "Europe/Amsterdam"`.
2. Press **Ctrl+A**, then **Ctrl+C**.
3. Go back to the **Apps Script editor** and click **`appsscript.json`** in the Files list on the left.
4. Click inside the code area, press **Ctrl+A**, then **Ctrl+V**.
5. Press **Ctrl+S** to save.

**Check:** the text in `appsscript.json` should now contain the words `Tasks` and `Calendar`.

### Step 6: Reload the spreadsheet

1. Close the Apps Script tab (everything is saved).
2. Go to the spreadsheet tab and **reload the page** (press **F5**, or click the circular arrow ⟳ next to the address bar).
3. Wait a few seconds. A new menu, **Course tools**, appears at the top, to the right of **Help**.

If you don't see it after 10 seconds, reload the page once more.

✅ **Part 1 is done.**

---

## Part 2: Give permission (one time only)

The first time you use any item in the **Course tools** menu, Google asks you to allow the helper to work with your spreadsheet, your Tasks and your Calendar.
This is normal. It happens **only once**.

1. Click on any **course tab** at the bottom of the spreadsheet (for example, a tab named after a course, **not** "Action Tasks" or "Assistants Procedure").
2. Click **Course tools → Preview (no changes)**.
3. A box says **Authorization required**. Click **Continue** (or **Review permissions**).
4. A new window opens and asks you to choose an account. **Click your own Google account.**
5. You may see a screen that says **"Google hasn't verified this app"**.
   This looks scary, but it is expected. Here is why: this helper was written just for you, and it is not a public app, so Google never reviewed it.
   It runs only inside your own account, and your data is not sent anywhere else.
   - Click the small **Advanced** link at the bottom left.
   - Then click **Go to Course tools (unsafe)**.
6. You now see a list of what the helper wants to do. It is short:

   | What Google says (the exact words may differ a little) | Why the helper needs it |
   |---|---|
   | See, edit, create and delete **this** spreadsheet | To read the course dates, and to write the trainer email and calendar link |
   | Display and run third-party web content in prompts and sidebars | To show you the small message boxes |
   | Create, edit, organize and delete all your **tasks** | To create the Tasks list |
   | See, edit, share and delete all the **calendars** you can access | To create the course calendar and share it with the trainer |

   **If there are checkboxes next to the items, tick "Select all".** If one item is left unticked, the helper cannot work.
7. Click **Continue** (or **Allow**).

The window closes. You may need to click **Course tools → Preview (no changes)** one more time.
A box appears with the list of tasks and dates of that course. Nothing has been created yet; this was only a preview.

> If nothing happens when you click Continue, your browser may have blocked the pop-up window.
> Look for a small blocked-window icon at the right end of the address bar, click it, choose **Always allow pop-ups**, and try again.

✅ **Setup is complete.** You will not need to do Part 1 or Part 2 again.

---

## Part 3: Everyday use: setting up a new course

Do this every time you prepare a new course.

### Step 1: Prepare the course tab, as you do today

Create the course tab the way you always do (for example, by duplicating an existing course tab), give it the course name, and fill in:

- **B1**: the course start date (next to "Course start")
- **B2**: the course end date (next to "Course end")

The dates in the **Exact Date** column update by themselves, as they always did.

> ⚠️ **If you duplicated a tab from another course:** look at cells **H1 and I1**.
> If I1 contains the **previous trainer's email**, change it to the new trainer's email, or delete both H1 and I1.
> Otherwise the new calendar will be shared with the wrong person.
> (The old calendar link in I2 is fine to leave; it is replaced automatically when you create the new calendar.)

**The tab name matters:** the Tasks list and the calendar get the same name as the tab. Pick the final name before you continue.

### Step 2: Preview

1. Click the course tab so it is the one you are looking at.
2. Click **Course tools → Preview (no changes)**.
3. Read the list. Each line shows who the task belongs to, the date, and the task:
   - `[PA]` = your task
   - `[Trainer]` = the trainer's task (the yellow rows)

   At the bottom you may see **Skipped**: those are rows without a date (for example, "Invite people who declared interest"). They are left out on purpose.
4. If a date looks wrong, fix it in the sheet and preview again.

### Step 3: Create everything

1. Click **Course tools → Create both**.
2. The first box confirms the **Tasks list**. Click **OK**.
3. If the tab has no trainer email yet, a box asks for it. Type the trainer's email address and click **OK**.
   (If you leave it empty and click OK, the calendar is created but not shared with anyone.)
4. Wait a few seconds. A box confirms the **calendar** and says who it was shared with. Click **OK**.

That's it. 🎉

### Where to find the results

- **The Tasks list**: open [Google Tasks](https://tasks.google.com), or click the **Tasks icon** ✔️ on the right side of Gmail or Google Calendar.
  Choose the list with the course name. On a phone, use the Google Tasks app.
- **The calendar**: open [Google Calendar](https://calendar.google.com). On the left, under **My calendars**, there is a new calendar with the course name.
  The link to it is also written in the course tab, in cell **I2** (or I1).
- **The trainer** gets an email from Google saying the calendar was shared with them. They click the link in that email to add it to their own calendar.

---

## Good to know

### Changing something after you created it

If you change a date, a task, or the trainer's email in the tab, just run **Course tools → Create both** again.

- **Tasks:** the helper asks *"Delete it and create it again from the sheet?"*. Click **Yes**.
  The old list is deleted and a new, correct one is made. (Any task you already ticked as done in that list is lost, so it is best to do this before you start ticking.)
- **Calendar:** the helper removes only the events **it** created, and makes them again with the new dates.
  Anything you added to that calendar by hand stays.

You can also use **Create Google Tasks list** or **Create trainer calendar** on their own, if you only want to update one of them.

### What the helper writes into the tab

The helper writes into two cells in the top rows of the course tab, on the right (usually columns H and I):

| Cell | Content |
|---|---|
| H1 / I1 | `Trainer email` and the address |
| H2 / I2 | `Calendar` and a link to the course calendar |

You can type the trainer's email into I1 yourself in advance (with the words `Trainer email` in H1). Then the helper will not ask for it.

### Renaming a tab

If you rename a course tab **after** creating its Tasks list, the next run makes a new list with the new name, and the list with the old name stays.
Delete the old list by hand in Google Tasks. The calendar is not affected; the helper still recognizes it.

### Who can see what

- The **Tasks list** is private to your Google account. Google Tasks cannot be shared.
- The **calendar** belongs to you and is shared with the trainer only.
- The reminders are set up for **your** account. To be sure the trainer gets reminders too, see [For the trainer](#for-the-trainer).
- Everything is created in the account of **whoever clicks the menu**. So please run the helper only from your own account.

---

## For the trainer

*(You can forward this part to the trainers.)*

After you receive the email "... has shared a calendar with you", click the link in it to add the course calendar.

To get reminders before each deadline:

1. Open [Google Calendar](https://calendar.google.com) on a computer.
2. On the left, find the course calendar (under **Other calendars** or **My calendars**), move the mouse over it and click the three dots **⋮** → **Settings and sharing**.
3. Scroll down to **All-day event notifications** and click **Add notification**.
4. Choose, for example: **Email**, **3 days before at 9am**. Add a second one for **1 day before** if you like.

---

## If something goes wrong

| What you see | What it means and what to do |
|---|---|
| There is no **Course tools** menu | Reload the spreadsheet page and wait 10 seconds. If it is still missing, open **Extensions → Apps Script** and check that **Code.gs** contains the long script and was saved (Part 1, Step 3). |
| `Cell next to "Course start" must be a date. Is this a course sheet?` | You are on a tab that is not a course, such as "Action Tasks" or "Assistants Procedure". Click on a course tab and try again. If you **are** on a course tab, check that **B1** holds a real date, next to the words "Course start" in **A1**. |
| `Tasks is not defined` or `Calendar is not defined` | The manifest was not pasted. Do **Part 1, Step 4 and Step 5**, then try again. |
| `Authorization is required to perform that action` | Permission was not given, or not all boxes were ticked. Click the menu item again and follow **Part 2**, ticking **Select all**. |
| `No dated tasks found on this sheet` | None of the rows have a date in the **Exact Date** column. Check that B1 has the start date. |
| A task appears one day off | Check the spreadsheet's time zone: **File → Settings → Time zone** should be your own time zone (for example, Amsterdam). Then run the helper again. |
| `This app is blocked` | Your Google account is managed by an organization that doesn't allow this. Ask Yoram or Ori. |
| Anything else | Take a screenshot of the message and send it to Ori. |

---

## Updating the helper later

If Ori tells you there is a new version:

1. Open the spreadsheet → **Extensions → Apps Script**.
2. Click **Code.gs**, and repeat **Part 1, Step 3** (copy from the link, select all, paste, save).
3. Only if Ori says the manifest changed: click **appsscript.json** and repeat **Part 1, Step 5**.
4. Reload the spreadsheet.

You usually won't need to give permission again.

---

## Questions

For anything unclear, contact **Ori** (Yoram's brother), who wrote the helper. Thank you for trying it! 🙏

---

<sub>For developers: the script is [`apps_script/Code.gs`](apps_script/Code.gs) (settings are in `CONFIG` at the top), and the manifest is [`apps_script/appsscript.json`](apps_script/appsscript.json).</sub>
