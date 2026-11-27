// How long a post, a comment or a message may be. The box itself never cuts what is typed or
// pasted into it: words that are too long stay whole, and the writer is told by how much, so
// nothing disappears without them knowing.
export const MAX_TEXT = 50000;

const count = (n) => n.toLocaleString('he-IL');

export function keepWhole(box, max = MAX_TEXT) {
  box.addEventListener('input', () => {
    const over = box.value.trim().length - max;
    box.setCustomValidity(over > 0
      ? `הטקסט ארוך מדי ב-${count(over)} תווים (אפשר עד ${count(max)}). שום דבר לא נחתך, אבל צריך לקצר לפני השליחה.` : '');
    if (over > 0) box.reportValidity();
  });
  return box;
}
