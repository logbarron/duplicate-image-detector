# **Complete Dark Mode Toggle Implementation Plan**

## **Overview**
Convert the existing system preference-based dark mode to a manual toggle system like Anthropic's, while maintaining backward compatibility.

## **Current State Analysis**
- ✅ Dark mode CSS exists using `@media (prefers-color-scheme: dark)`
- ✅ CSS variables are well-organized
- ✅ Clean HTML structure for adding toggle button
- ✅ All dark mode colors already defined

## **Target State**
- Manual toggle button in the header (sun/moon icons)
- User preference saved in localStorage
- Fallback to system preference if no user preference set
- Smooth transitions between modes

---

## **Step-by-Step Implementation Plan**

### **Step 1: Add Toggle Button to HTML**
**Location:** Insert into `.content-header` section (around line 225)
**What:** Add button with sun/moon SVG icons exactly like Anthropic's

```html
<!-- Add this inside .content-header -->
<button id="darkModeToggle" class="theme-toggle" aria-label="Toggle dark mode">
  <!-- Sun icon (visible in light mode) -->
  <svg class="sun-icon" width="16" height="16" viewBox="0 0 16 16">...</svg>
  <!-- Moon icon (visible in dark mode) -->
  <svg class="moon-icon" width="16" height="16" viewBox="0 0 16 16">...</svg>
</button>
```

### **Step 2: Add Toggle Button CSS**
**Location:** Add after existing button styles (around line 530)
**What:** Style the toggle button and icon visibility

```css
.theme-toggle {
  padding: 8px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
  cursor: pointer;
  transition: all 0.15s var(--transition);
}

.theme-toggle:hover {
  background: var(--surface-2);
}

.sun-icon { display: block; }
.moon-icon { display: none; }

html.dark .sun-icon { display: none; }
html.dark .moon-icon { display: block; }
```

### **Step 3: Convert Dark Mode CSS**
**Location:** Replace `@media (prefers-color-scheme: dark)` blocks (lines 38-50 and 301-305)
**What:** Change from media query to class-based system

```css
/* FROM: */
@media (prefers-color-scheme: dark) { :root { ... } }

/* TO: */
html.dark { ... }
```

### **Step 4: Add JavaScript Toggle Logic**
**Location:** Add to `<script>` section (around line 876)
**What:** Toggle functionality with localStorage support

```javascript
// Dark mode toggle functionality
function initDarkMode() {
  const savedMode = localStorage.getItem('darkMode');
  const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  
  if (savedMode === 'dark' || (!savedMode && systemPrefersDark)) {
    document.documentElement.classList.add('dark');
  }
}

function toggleDarkMode() {
  const isDark = document.documentElement.classList.toggle('dark');
  localStorage.setItem('darkMode', isDark ? 'dark' : 'light');
}

// Initialize on page load
initDarkMode();

// Add event listener
document.getElementById('darkModeToggle').addEventListener('click', toggleDarkMode);
```

---

## **Detailed File Changes**

### **Files to Modify:** 
- `duplicate-detector-ui.html` (only file that needs changes)

### **Lines to Change:**
- **Line ~230:** Add toggle button to header
- **Line ~38-50:** Convert first dark mode CSS block
- **Line ~301-305:** Convert second dark mode CSS block  
- **Line ~530:** Add toggle button CSS
- **Line ~876:** Add JavaScript toggle logic

### **Estimated Changes:**
- **Add:** ~40 lines (button HTML + CSS + JavaScript)
- **Modify:** ~15 lines (CSS media queries → class selectors)
- **Total:** ~55 lines changed/added

---

## **Testing Plan**

1. **Visual Test:** Toggle button appears in header
2. **Functionality Test:** Clicking toggles between light/dark
3. **Persistence Test:** Reload page, mode is remembered
4. **System Preference Test:** New users get system preference
5. **Icon Test:** Sun shows in light mode, moon in dark mode

---

## **Risk Assessment: LOW** 🟢

- **Backward compatibility:** Maintained (system preference still works for new users)
- **No breaking changes:** Only additions and CSS selector changes
- **Fallback:** If JavaScript fails, defaults to system preference
- **Isolated changes:** All changes in one file

---

## **Implementation Notes**

### **Anthropic's Button Structure (Reference)**
```html
<button class="group p-2 flex items-center justify-center" aria-label="Toggle dark mode">
  <svg class="h-4 w-4 block text-gray-400 dark:hidden group-hover:text-gray-600">
    <!-- Sun icon SVG -->
  </svg>
  <svg class="h-4 w-4 hidden dark:block text-gray-500 dark:group-hover:text-gray-300">
    <!-- Moon icon SVG -->
  </svg>
</button>
```

### **CSS Pattern (Reference)**
```css
html.dark .token { /* dark mode styles */ }
html:not(.dark) .codeblock-light .token { /* light mode styles */ }
```

---

**Ready to proceed?** This should take about 5-10 minutes to implement. Each step is clearly defined and low-risk.