import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { Contact } from '../services/contactservice/contact';
import { ContactMessage } from '../models/contact.model';

@Component({
  selector: 'app-contact',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule],
  templateUrl: './contact.component.html',
  styleUrls: ['./contact.component.css']
})
export class ContactComponent {
  contactForm: FormGroup;
  loading = false;
  showSuccess = false;
  showError = false;
  submitted = false;

  constructor(
    private fb: FormBuilder,
    private contactService: Contact
  ) {
    this.contactForm = this.fb.group({
      name: ['', [Validators.required, Validators.minLength(2)]],
      email: ['', [Validators.required, Validators.email]],
      subject: ['', [Validators.required, Validators.minLength(5)]],
      message: ['', [Validators.required, Validators.minLength(10)]]
    });
  }

  get f() {
    return this.contactForm.controls;
  }

  onSubmit() {
    this.submitted = true;
    if (this.contactForm.invalid) return;

    this.loading = true;
    this.showSuccess = false;
    this.showError = false;

    const payload: ContactMessage = this.contactForm.value;

    this.contactService.sendMessage(payload).subscribe({
      next: (response) => {
        console.log('Message envoyé !', response);
        this.loading = false;
        this.showSuccess = true;
        this.contactForm.reset();
        this.submitted = false;
        setTimeout(() => (this.showSuccess = false), 5000);
      },
      error: (err) => {
        console.error('Erreur API:', err);
        this.loading = false;
        this.showError = true;
        setTimeout(() => (this.showError = false), 5000);
      }
    });
  }
}
