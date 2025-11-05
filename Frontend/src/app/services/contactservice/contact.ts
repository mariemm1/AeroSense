// src/app/services/contact.service.ts
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { ContactMessage } from '../../models/contact.model';

@Injectable({
  providedIn: 'root'
})
export class Contact {
  private apiUrl = 'http://127.0.0.1:8000/api/contact/';

  constructor(private http: HttpClient) {}

  sendMessage(message: ContactMessage): Observable<any> {
    return this.http.post(this.apiUrl, message);
  }
}